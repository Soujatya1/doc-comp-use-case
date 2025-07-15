import streamlit as st
import pandas as pd
import fitz
import io
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
import json
import re
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentComparer:
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str = "2024-02-01", 
                 deployment_name: str = "gpt-4"):
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=deployment_name,
            temperature=0.1,
            max_tokens=4000
        )

        self.target_sections = [
            "FORWARDING LETTER",
            "PREAMBLE",
            "SCHEDULE",
            "DEFINITIONS & ABBREVIATIONS",
            "PART C",
            "PART D",
            "PART F",
            "PART G",
            "ANNEXURE AA",
            "ANNEXURE BB",
            "ANNEXURE CC"
        ]
        
        self.section_headers = [
            "FORWARDING LETTER",
            "PREAMBLE",
            "SCHEDULE",
            "DEFINITIONS & ABBREVIATIONS",
            "PART C",
            "PART D",
            "PART F",
            "PART G",
            "ANNEXURE AA",
            "ANNEXURE BB",
            "ANNEXURE CC"
        ]
        
        self.exclude_sections = [
            "First Premium Receipt",
            "Your Renewal Page"
        ]
        
        self.max_input_chars = 12000
        self.chunk_size = 8000

    def clean_text_for_comparison(self, text: str) -> str:

        if not text or text == "NOT FOUND":
            return text
        
        cleaned_text = text
        
        # Only remove clearly placeholder content, not all angle brackets
        placeholder_patterns = [
            r'<\s*Name\s+of\s+the\s+Policyholder\s*>',
            r'<\s*Address\s+of\s+the\s+Policyholder\s*>',
            r'<\s*Mr\./Mrs\./Ms\.\s*>',
            r'<\s*x+\s*>',
            r'<\s*X+\s*>',
            r'<\s*[Dd]ate\s*>',
            r'<\s*[Aa]mount\s*>',
            r'<\s*[Nn]umber\s*>',
        ]
        
        for pattern in placeholder_patterns:
            cleaned_text = re.sub(pattern, '[PLACEHOLDER]', cleaned_text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        
        return cleaned_text.strip()

    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text("text") + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def extract_sample_number_from_filename(self, filename: str) -> str:
        patterns = [
            r'(\d{10})__Sample_\d+_',
            r'(\d{8,})__Sample',
            r'^(\d{8,})',
        ]
    
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1)
    
        return "001"

    def is_image_only_page(self, page):
        text = page.get_text("text").strip()
        return not text and (page.get_images(full=True) or page.get_drawings())

    def extract_final_filtered_pdf(self, uploaded_file_bytes, is_customer_copy=False):
        doc = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
        filtered_doc = fitz.open()
        text_summary = ""
        debug_logs = ""
        current_section = "UNKNOWN"
 
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            text_lower = text.lower()
            log = f"Page {page_num + 1}: "
 
            if self.is_image_only_page(page):
                debug_logs += log + " Skipped (Image-only page)\n"
                continue
 
            if any(section.lower() in text_lower for section in self.exclude_sections):
                debug_logs += log + "Skipped (Excluded section)\n"
                continue
 
            for header in self.section_headers:
                if header.lower() in text_lower:
                    current_section = header
                    break
 
            debug_logs += log + f"Included (Section: {current_section})\n"
 
            section_label = f"\n--- {current_section} ---"
            page_marker = f"\n--- Page {page_num + 1} ---"
            filtered_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            text_summary += f"{section_label}{page_marker}\n{text.strip()}\n"
 
        output_stream = io.BytesIO()
        if len(filtered_doc) > 0:
            filtered_doc.save(output_stream)
            output_stream.seek(0)
            return text_summary.strip(), output_stream
        else:
            return "", None

    def is_header_line(self, line):
        return line.strip().upper() in self.section_headers

    def extract_forwarding_letter_section_filed(self, text):
        lines = text.splitlines()
        try:
            start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "FORWARDING LETTER")
            end_idx = next(i for i, line in enumerate(lines[start_idx+1:], start=start_idx+1)
                           if self.is_header_line(line) and line.strip().upper() != "FORWARDING LETTER")
            return "\n".join(lines[start_idx:end_idx]).strip()
        except StopIteration:
            return "FORWARDING LETTER section not found in Filed Copy."

    def extract_forwarding_letter_section_customer(self, text):
        lines = text.splitlines()
        try:
            end_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "PREAMBLE")
            return "\n".join(lines[:end_idx]).strip()
        except StopIteration:
            return "FORWARDING LETTER section not found in Customer Copy."

    def extract_schedule_section(self, text):
        lines = text.splitlines()
        try:
            preamble_start = next(i for i, line in enumerate(lines) if line.strip().upper() == "PREAMBLE")
            schedule_start = None
            for j in range(preamble_start + 1, len(lines)):
                if lines[j].strip().upper() == "SCHEDULE":
                    schedule_start = j + 1
                    break
            if schedule_start is None:
                return "SCHEDULE header not found after PREAMBLE."
        except StopIteration:
            return "PREAMBLE not found. Cannot extract SCHEDULE section."

        end_idx = None
        for j in range(schedule_start, len(lines)):
            line_upper = lines[j].strip().upper()
            if "DETAILS OF THE NOMINEE" in line_upper:
                end_idx = j
                break
            if line_upper in self.section_headers and line_upper != "SCHEDULE":
                end_idx = j
                break

        end_idx = end_idx or len(lines)
        return "\n".join(lines[schedule_start:end_idx]).strip()

    def extract_annexure_cc_section(self, text):
        lines = text.splitlines()
        try:
            annexure_cc_start = None
            for i, line in enumerate(lines):
                if line.strip().upper() == "ANNEXURE CC":
                    annexure_cc_start = i + 1
                    break
        
            if annexure_cc_start is None:
                return "ANNEXURE CC header not found."
        except StopIteration:
            return "ANNEXURE CC not found."

        end_idx = None
        for j in range(annexure_cc_start, len(lines)):
            line_upper = lines[j].strip().upper()
            if "APPLICATION / PROPOSAL NO. WITH BARCODE" in line_upper:
                end_idx = j
                break
    
        end_idx = end_idx or len(lines)
        return "\n".join(lines[annexure_cc_start:end_idx]).strip()

    def extract_sections(self, text):
        lines = text.splitlines()
        sections = {}
        header_positions = []

        for i, line in enumerate(lines):
            if self.is_header_line(line):
                matched_header = line.strip().upper()
                header_positions.append((matched_header, i))

        for idx, (header, start_idx) in enumerate(header_positions):
            if header in ["FORWARDING LETTER", "SCHEDULE", "ANNEXURE CC"]:
                continue
            end_idx = header_positions[idx + 1][1] if idx + 1 < len(header_positions) else len(lines)
            content = "\n".join(lines[start_idx:end_idx]).strip()
            sections[header] = content

        return sections

    def filter_sections_with_rules(self, document_text: str, doc_name: str, is_customer_copy: bool = False) -> Dict[str, str]:
        try:
            st.info(f"Extracting sections from {doc_name} using rule-based approach...")
            
            sections = self.extract_sections(document_text)
            
            if is_customer_copy:
                sections["FORWARDING LETTER"] = self.extract_forwarding_letter_section_customer(document_text)
            else:
                sections["FORWARDING LETTER"] = self.extract_forwarding_letter_section_filed(document_text)
            
            sections["SCHEDULE"] = self.extract_schedule_section(document_text)
            sections["ANNEXURE CC"] = self.extract_annexure_cc_section(document_text)
            
            filtered_sections = {}
            for section in self.target_sections:
                if section in sections:
                    filtered_sections[section] = sections[section]
                else:
                    filtered_sections[section] = "NOT FOUND"
            
            st.success(f"Successfully extracted sections from {doc_name}: {list(filtered_sections.keys())}")
            return filtered_sections
            
        except Exception as e:
            logger.error(f"Error in rule-based section extraction for {doc_name}: {str(e)}")
            return {section: "NOT FOUND" for section in self.target_sections}

    def compare_documents_with_llm(self, doc1_sections: Dict[str, str], doc2_sections: Dict[str, str], 
                           doc1_name: str, doc2_name: str, sample_number: str) -> List[Dict]:

        comparison_results = []

        for section in self.target_sections:
            doc1_content = doc1_sections.get(section, "NOT FOUND")
            doc2_content = doc2_sections.get(section, "NOT FOUND")

            doc1_cleaned = self.clean_text_for_comparison(doc1_content)
            doc2_cleaned = self.clean_text_for_comparison(doc2_content)

            total_content_size = len(doc1_cleaned) + len(doc2_cleaned)
        
            if total_content_size > 12000:
                doc1_cleaned_truncated = doc1_cleaned[:10000] if doc1_cleaned != "NOT FOUND" else "NOT FOUND"
                doc2_cleaned_truncated = doc2_cleaned[:10000] if doc2_cleaned != "NOT FOUND" else "NOT FOUND"
            
                st.warning(f"Section {section} content truncated for comparison due to size")
            
                comparison_result = self._compare_section_content(
                    doc1_cleaned_truncated, doc2_cleaned_truncated, section, sample_number, 
                    doc1_content, doc2_content
                )
            else:
                comparison_result = self._compare_section_content(
                    doc1_cleaned, doc2_cleaned, section, sample_number,
                    doc1_content, doc2_content
                )
        
            comparison_results.append(comparison_result)

        return comparison_results

    def create_improved_system_prompt(self, section: str, doc1_cleaned: str, doc2_cleaned: str) -> str:
    """
    Creates an improved system prompt for document comparison with better instructions
    and section-specific guidance.
    """
    
    # Section-specific focus areas
    section_guidance = {
        "FORWARDING LETTER": """
        Focus specifically on differences in:
        - Required document lists (additions, removals, or changes)
        - Free-look period duration and conditions
        - Cancellation procedures and timelines
        - Regulatory references (Section 45 of Insurance Act, IRDAI regulations)
        - Contact information for policy servicing
        - Instructions for policy issuance or modifications
        - Refund conditions and procedures
        """,
        "SCHEDULE": """
        Focus specifically on differences in:
        - Policy terms and durations
        - Premium amounts and payment frequencies
        - Coverage limits and benefit amounts
        - Policy effective dates and renewal terms
        - Rider information and additional benefits
        - Sum assured or coverage amounts
        """,
        "PREAMBLE": """
        Focus specifically on differences in:
        - Legal declarations and statements
        - Regulatory compliance language
        - Policy introduction and purpose
        - Legal framework references
        - Contractual terms and conditions
        """,
        "DEFINITIONS & ABBREVIATIONS": """
        Focus specifically on differences in:
        - Definition changes or additions
        - New abbreviations or modifications
        - Terminology clarifications
        - Legal or technical term explanations
        """,
        "PART C": """
        Focus specifically on differences in:
        - Benefit descriptions and conditions
        - Eligibility criteria
        - Exclusions and limitations
        - Claim procedures
        """,
        "PART D": """
        Focus specifically on differences in:
        - Premium payment terms
        - Grace period provisions
        - Lapse and revival conditions
        - Premium calculation methods
        """,
        "PART F": """
        Focus specifically on differences in:
        - Claim settlement procedures
        - Required documentation for claims
        - Claim processing timelines
        - Dispute resolution mechanisms
        """,
        "PART G": """
        Focus specifically on differences in:
        - General conditions and provisions
        - Policy administration procedures
        - Regulatory compliance requirements
        - Miscellaneous terms and conditions
        """,
        "ANNEXURE AA": """
        Focus specifically on differences in:
        - Specific annexure content
        - Additional terms and conditions
        - Supplementary information
        """,
        "ANNEXURE BB": """
        Focus specifically on differences in:
        - Specific annexure content
        - Additional terms and conditions
        - Supplementary information
        """,
        "ANNEXURE CC": """
        Focus specifically on differences in:
        - Specific annexure content
        - Additional terms and conditions
        - Supplementary information
        """
    }
    
    specific_guidance = section_guidance.get(section, """
    Focus on all meaningful content differences including:
    - Changes in terms, conditions, or procedures
    - Addition or removal of clauses
    - Modifications in amounts, percentages, or timeframes
    """)
    
    return f"""You are an expert document comparison analyst specializing in insurance policy documents. Your task is to perform a precise comparison of two versions of the same document section.

SECTION BEING ANALYZED: {section}

SPECIFIC ANALYSIS FOCUS:
{specific_guidance}

COMPARISON INSTRUCTIONS:

1. WHAT TO IDENTIFY AS MEANINGFUL DIFFERENCES:
   ✓ Changes in policy terms, conditions, or procedures
   ✓ Addition or removal of clauses, requirements, or benefits
   ✓ Modifications in amounts, percentages, timeframes, or durations
   ✓ Changes in contact information, addresses, or service procedures
   ✓ Alterations in regulatory references or compliance requirements
   ✓ Different document requirements or submission procedures
   ✓ Changes in legal language that affect policy interpretation

2. WHAT TO IGNORE (NOT meaningful differences):
   ✗ Personal names (policyholder names, beneficiary names)
   ✗ Policy numbers, application numbers, certificate numbers
   ✗ Personal identification numbers (Aadhaar, PAN, etc.)
   ✗ Personal addresses, phone numbers, email addresses
   ✗ Dates that are clearly personalized (policy issue dates, birth dates)
   ✗ Formatting differences (spacing, font, alignment)
   ✗ Placeholder text like <Name>, <Address>, <Number>
   ✗ Minor grammatical or spelling corrections that don't change meaning

3. OUTPUT FORMAT:
   If you find meaningful differences, present them as a clear, numbered list:
   
   1. [Specific description of difference]
   2. [Another difference if found]
   3. [Continue numbering for each difference]

   If NO meaningful differences exist, respond with exactly:
   "NO_MEANINGFUL_DIFFERENCES"

4. EXAMPLES OF GOOD DIFFERENCE DESCRIPTIONS:
   ✓ "Free-look period changed from 15 days in Filed Copy to 30 days in Customer Copy"
   ✓ "Additional document requirement added in Customer Copy: PAN card photocopy"
   ✓ "Contact email for policy servicing changed from service@oldcompany.com to help@newcompany.com"
   ✓ "Section 45 reference removed from Customer Copy"
   ✓ "Premium payment grace period extended from 30 days to 45 days in Customer Copy"

5. EXAMPLES OF DIFFERENCES TO IGNORE:
   ✗ "Policyholder name changed from John Smith to Jane Doe"
   ✗ "Policy number changed from POL123456 to POL789012"
   ✗ "Different formatting in address layout"

DOCUMENTS TO COMPARE:

FILED COPY CONTENT:
{doc1_cleaned}

CUSTOMER COPY CONTENT:
{doc2_cleaned}

ANALYSIS INSTRUCTION:
Analyze both documents carefully and identify only meaningful content differences as defined above. Be precise and specific in your descriptions. If no meaningful differences exist, respond with "NO_MEANINGFUL_DIFFERENCES"."""
    
    def _compare_section_content(self, doc1_cleaned: str, doc2_cleaned: str, section: str, 
                           sample_number: str, doc1_original: str = None, 
                           doc2_original: str = None) -> Dict:
    
        display_doc1 = doc1_original if doc1_original is not None else doc1_cleaned
        display_doc2 = doc2_original if doc2_original is not None else doc2_cleaned

        system_prompt = self.create_improved_system_prompt(section, doc1_cleaned, doc2_cleaned)
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please provide your analysis.")
            ]

            response = self.llm.invoke(messages)

            comparison_result = self._create_section_result(
                response.content, section, display_doc1, display_doc2, sample_number
            )
            return comparison_result

        except Exception as e:
            logger.error(f"Error comparing section {section}: {str(e)}")
            return {
                'Samples affected': sample_number,
                'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
                'Page': self._get_page_name(section),
                'Sub-category of Observation': f'Error during comparison: {str(e)}',
                'Content': f"Filed Copy: {display_doc1[:500]}{'...' if len(display_doc1) > 500 else ''}\n\n"
                       f"Customer Copy: {display_doc2[:500]}{'...' if len(display_doc2) > 500 else ''}"
            }
    
    def _get_page_name(self, section: str) -> str:
        section_mapping = {
            "FORWARDING LETTER": "Forwarding letter",
            "PREAMBLE": "Preamble", 
            "SCHEDULE": "Schedule",
            "DEFINITIONS & ABBREVIATIONS": "Definitions and Abbreviations"
        }
        return section_mapping.get(section, section.title())
    
    def _create_section_result(self, response_content: str, section: str, doc1_content: str, 
                             doc2_content: str, sample_number: str) -> Dict:
        
        no_diff_indicators = [
            "NO_CONTENT_DIFFERENCE",
            "NO MEANINGFUL CONTENT DIFFERENCES",
            "NO SUBSTANTIVE DIFFERENCES",
            "CONTENT IS IDENTICAL",
            "NO CONTENT CHANGES",
            "SAME CONTENT",
            "IDENTICAL CONTENT"
        ]
        
        response_upper = response_content.upper()
        has_differences = not any(indicator in response_upper for indicator in no_diff_indicators)
        
        formatting_only_indicators = [
            "ONLY FORMATTING DIFFERENCES",
            "ONLY STRUCTURAL DIFFERENCES", 
            "ONLY LAYOUT DIFFERENCES",
            "SAME CONTENT, DIFFERENT FORMAT",
            "FORMATTING VARIATIONS ONLY",
            "STRUCTURAL CHANGES ONLY"
        ]
        
        if any(indicator in response_upper for indicator in formatting_only_indicators):
            has_differences = False
        
        content_display = self._prepare_content_display(doc1_content, doc2_content, section)
        
        if doc1_content == "NOT FOUND" and doc2_content == "NOT FOUND":
            return {
                'Samples affected': sample_number,
                'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
                'Page': self._get_page_name(section),
                'Sub-category of Observation': 'Section not found in both documents',
                'Content': 'Section not found in either document'
            }
        elif doc1_content == "NOT FOUND":
            return {
                'Samples affected': sample_number,
                'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
                'Page': self._get_page_name(section),
                'Sub-category of Observation': 'Section missing in Filed Copy',
                'Content': content_display
            }
        elif doc2_content == "NOT FOUND":
            return {
                'Samples affected': sample_number,
                'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
                'Page': self._get_page_name(section),
                'Sub-category of Observation': 'Section missing in Customer Copy',
                'Content': content_display
            }
        
        if has_differences:
            sub_category = self._parse_difference_description(response_content, section)
            observation_category = 'Mismatch of content between Filed Copy and customer copy'
        else:
            sub_category = 'No meaningful content differences found'
            observation_category = 'Content matches between Filed Copy and customer copy'
        
        return {
            'Samples affected': sample_number,
            'Observation - Category': observation_category,
            'Page': self._get_page_name(section),
            'Sub-category of Observation': sub_category,
            'Content': content_display
        }
    
    def _prepare_content_display(self, doc1_content: str, doc2_content: str, section: str) -> str:
        
        if doc1_content == doc2_content and doc1_content != "NOT FOUND":
            content = doc1_content
            if len(content) > 1000:
                content = content[:1000] + "... [Content truncated for display]"
            return f"[IDENTICAL CONTENT]\n{content}"
        
        result_parts = []
        
        if doc1_content != "NOT FOUND":
            filed_content = doc1_content
            if len(filed_content) > 500:
                filed_content = filed_content[:500] + "... [Truncated]"
            result_parts.append(f"[FILED COPY]\n{filed_content}")
        
        if doc2_content != "NOT FOUND":
            customer_content = doc2_content
            if len(customer_content) > 500:
                customer_content = customer_content[:500] + "... [Truncated]"
            result_parts.append(f"[CUSTOMER COPY]\n{customer_content}")
        
        return "\n\n".join(result_parts)
    
    def _parse_difference_description(self, response_content: str, section: str) -> str:

        sub_category = response_content.strip()

        prefixes_to_remove = [
            "Meaningful differences found:",
            "Content differences identified:",
            "The following content differences were found:",
            "Key content differences:",
            "Analysis shows the following content changes:",
            "Comparison reveals content differences:",
            "The following meaningful content differences were identified:",
            "Content analysis reveals:",
            "Substantive differences:",
            "Differences found:",
            "The differences are:",
            "Key differences:",
            "Analysis shows:",
            "Comparison reveals:",
            "The following differences were identified:"
        ]

        for prefix in prefixes_to_remove:
            if sub_category.upper().startswith(prefix.upper()):
                sub_category = sub_category[len(prefix):].strip()
                break

        sub_category = re.sub(r'\s+', ' ', sub_category)
        sub_category = sub_category.strip('. \n\r\t')

        sub_category = re.sub(r'^[-•*]\s*', '', sub_category)
        sub_category = re.sub(r'^\d+\.\s*', '', sub_category)

        lines = [line.strip() for line in sub_category.split('\n') if line.strip()]
        if lines:
            for line in lines:
                if len(line) > 10:
                    sub_category = line
                    break
            else:
                sub_category = lines[0]

        if sub_category and len(sub_category) > 0:
            sub_category = sub_category[0].upper() + sub_category[1:]

        if len(sub_category) < 10:
            sub_category = f"Meaningful content differences identified in section {section}"

        return sub_category

    
    def create_excel_report(self, comparison_results: List[Dict], doc1_name: str, doc2_name: str) -> bytes:
    
        df = pd.DataFrame(comparison_results)
    
        if not df.empty:
            df = df[['Samples affected', 'Observation - Category', 'Page', 'Sub-category of Observation']]
    
        output = io.BytesIO()
    
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Document_Comparison', index=False)
        
            if not df.empty:
                workbook = writer.book
                worksheet = writer.sheets['Document_Comparison']
            
                from openpyxl.styles import Alignment
            
                total_rows = len(df)
                if total_rows > 1:
                    start_row = 2
                    end_row = total_rows + 1
                
                    merge_range = f'B{start_row}:B{end_row}'
                    worksheet.merge_cells(merge_range)
                
                    merged_cell = worksheet[f'B{start_row}']
                    merged_cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value:
                                if column_letter == 'D':
                                    max_length = max(max_length, min(len(str(cell.value)), 80))
                                elif column_letter == 'E':
                                    max_length = max(max_length, min(len(str(cell.value)), 120))
                                else:
                                    max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    
                    if column_letter == 'D':
                        adjusted_width = min(max_length + 2, 80)
                    elif column_letter == 'E':
                        adjusted_width = min(max_length + 2, 120)
                    else:
                        adjusted_width = min(max_length + 2, 30)
                    
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                for row in worksheet.iter_rows():
                    for cell in row:
                        if not cell.coordinate.startswith('B') or cell.row == 1:
                            cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            sections_with_differences = len([r for r in comparison_results if 'Mismatch' in r.get('Observation - Category', '')])
            sections_without_differences = len(comparison_results) - sections_with_differences
        
        output.seek(0)
        return output.getvalue()

def main():
    st.set_page_config(
        page_title="Enhanced Document Comparer with Azure OpenAI",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Enhanced Document Comparer with Azure OpenAI")
    st.markdown("Upload two PDF documents to compare specific sections using rule-based extraction and Azure OpenAI-powered comparison")
    
    with st.sidebar:
        st.header("🔧 Azure OpenAI Configuration")
        
        azure_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            placeholder="https://your-resource-name.openai.azure.com/",
            help="Your Azure OpenAI resource endpoint URL"
        )
        
        api_key = st.text_input(
            "Azure OpenAI API Key",
            placeholder="Enter your Azure OpenAI API key",
            type="password"
        )
        
        deployment_name = st.text_input(
            "Deployment Name",
            value="gpt-4",
            placeholder="Enter your deployment name",
            help="The name of your GPT model deployment in Azure"
        )
        
        api_version = st.selectbox(
            "API Version",
            options=["2025-01-01-preview"],
            index=0,
            help="Azure OpenAI API version"
        )
        
        st.markdown("---")
        st.markdown("**Target Sections:**")
        st.markdown("• Forwarding Letter")
        st.markdown("• Preamble")
        st.markdown("• Schedule")
        st.markdown("• Definitions & Abbreviations")
        
        st.markdown("---")
        st.markdown("**Analysis Features:**")
        st.markdown("✅ Rule-based section extraction")
        st.markdown("✅ Azure OpenAI-powered comparison")
        st.markdown("✅ Complete content display")
        st.markdown("✅ Structured Excel output")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Document 1 (Filed Copy)")
        doc1_file = st.file_uploader(
            "Upload first PDF document",
            type=['pdf'],
            key="doc1"
        )
        
        if doc1_file:
            st.success(f"✅ {doc1_file.name} uploaded successfully")
    
    with col2:
        st.subheader("📄 Document 2 (Customer Copy)")
        doc2_file = st.file_uploader(
            "Upload second PDF document",
            type=['pdf'],
            key="doc2"
        )
        
        if doc2_file:
            st.success(f"✅ {doc2_file.name} uploaded successfully")
    
    # Process documents
    if st.button("🔍 Analyze All Sections with Rule-based Extraction", type="primary"):
        if not azure_endpoint or not api_key or not deployment_name:
            st.error("❌ Please provide all Azure OpenAI configuration details")
            return
        
        if not doc1_file or not doc2_file:
            st.error("❌ Please upload both PDF documents")
            return
        
        try:
            comparer = DocumentComparer(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                deployment_name=deployment_name
            )
            
            with st.spinner("🔄 Processing documents..."):
                st.info("📖 Extracting text from documents...")
                doc1_text = comparer.extract_text_from_pdf(doc1_file)
                doc2_text = comparer.extract_text_from_pdf(doc2_file)
                
                if not doc1_text or not doc2_text:
                    st.error("❌ Failed to extract text from one or both documents")
                    return
                
                sample_number = comparer.extract_sample_number_from_filename(doc2_file.name)
                
                st.info("📋 Extracting sections using rule-based approach...")
                doc1_sections = comparer.filter_sections_with_rules(doc1_text, doc1_file.name, is_customer_copy=False)
                logger.info("📄 Filed Copy Extracted Sections:\n" + json.dumps(doc1_sections, indent=2))
                doc2_sections = comparer.filter_sections_with_rules(doc2_text, doc2_file.name, is_customer_copy=True)
                logger.info("📄 Customer Copy Extracted Sections:\n" + json.dumps(doc2_sections, indent=2))
                
                st.info("🔍 Analyzing sections with AI-powered comparison...")
                comparison_results = comparer.compare_documents_with_llm(
                    doc1_sections, doc2_sections, doc1_file.name, doc2_file.name, sample_number
                )
                
                st.info("📊 Generating comprehensive Excel report...")
                excel_data = comparer.create_excel_report(
                    comparison_results, doc1_file.name, doc2_file.name
                )
            
            st.success("✅ Complete document analysis with rule-based extraction completed successfully!")
            
            st.subheader("📊 Analysis Results")
            
            sections_with_differences = len([r for r in comparison_results if 'Mismatch' in r.get('Observation - Category', '')])
            sections_without_differences = len(comparison_results) - sections_with_differences
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Sections Analyzed", len(comparer.target_sections))
            
            with col2:
                st.metric("Sections with Differences", sections_with_differences)
            
            with col3:
                st.metric("Sections without Differences", sections_without_differences)
            
            with col4:
                st.metric("Sample Number", sample_number)
            
            st.subheader("📋 Complete Section Analysis")
            df_display = pd.DataFrame(comparison_results)
            
            df_display_short = df_display.copy()
            df_display_short['Content (Preview)'] = df_display_short['Content'].apply(
                lambda x: x[:200] + "..." if len(str(x)) > 200 else x
            )
            
            display_columns = ['Samples affected', 'Observation - Category', 'Page', 'Sub-category of Observation', 'Content (Preview)']
            st.dataframe(df_display_short[display_columns], use_container_width=True)
            
            st.info("ℹ️ **Note**: Complete content is available in the downloadable Excel report. Preview shows first 200 characters.")
            
            st.subheader("📥 Download Complete Report")
            st.download_button(
                label="📊 Download Complete Analysis with Content",
                data=excel_data,
                file_name=f"complete_analysis_{sample_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
