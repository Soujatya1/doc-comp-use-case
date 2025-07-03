import streamlit as st
import pandas as pd
import PyPDF2
import io
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentComparer:
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str, deployment_name: str):
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=deployment_name,
            temperature=0.1,
            max_tokens=4000
        )

        self.section_headers = [
            "FORWARDING LETTER",
            "PREAMBLE", 
            "SCHEDULE",
            "DEFINITIONS & ABBREVIATIONS"
        ]
        
        # Alternative headers for flexible matching
        self.header_variations = {
            "FORWARDING LETTER": [
                "FORWARDING LETTER", "COVERING LETTER", "TRANSMITTAL LETTER",
                "LETTER", "SUB:", "SUBJECT:", "REF:"
            ],
            "PREAMBLE": [
                "PREAMBLE", "WHEREAS", "NOW THEREFORE", "WITNESSETH"
            ],
            "SCHEDULE": [
                "SCHEDULE", "ANNEXURE", "ATTACHMENT", "EXHIBIT", 
                "DETAILS", "PARTICULARS"
            ],
            "DEFINITIONS & ABBREVIATIONS": [
                "DEFINITIONS & ABBREVIATIONS", "DEFINITIONS AND ABBREVIATIONS",
                "DEFINITIONS", "ABBREVIATIONS", "GLOSSARY", "TERMS"
            ]
        }

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def extract_sample_number_from_filename(self, filename: str) -> str:
        """Extract sample number from filename"""
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

    def normalize_text_for_matching(self, text: str) -> str:
        """Normalize text for better header matching"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s&:.]', ' ', text)
        return text.upper()

    def find_section_boundaries(self, text: str, doc_type: str = "filed") -> Dict[str, Tuple[int, int]]:
        """
        Find start and end positions of each section in the document
        Returns dict with section_name: (start_pos, end_pos)
        """
        normalized_text = self.normalize_text_for_matching(text)
        section_positions = {}
        
        # Find all potential header positions
        all_headers_found = []
        
        for section_name, variations in self.header_variations.items():
            for variation in variations:
                variation_normalized = self.normalize_text_for_matching(variation)
                
                # Find all occurrences of this variation
                start_pos = 0
                while True:
                    pos = normalized_text.find(variation_normalized, start_pos)
                    if pos == -1:
                        break
                    
                    # Check if it's likely a section header (at start of line or after significant whitespace)
                    if pos == 0 or normalized_text[pos-1] in ['\n', ' ']:
                        # Check if it's not just part of a larger word
                        end_pos = pos + len(variation_normalized)
                        if end_pos >= len(normalized_text) or normalized_text[end_pos] in ['\n', ' ', ':', '.']:
                            all_headers_found.append((pos, section_name, variation))
                    
                    start_pos = pos + 1
        
        # Sort headers by position
        all_headers_found.sort(key=lambda x: x[0])
        
        # Remove duplicates (keep the first occurrence of each section)
        seen_sections = set()
        unique_headers = []
        for pos, section_name, variation in all_headers_found:
            if section_name not in seen_sections:
                unique_headers.append((pos, section_name, variation))
                seen_sections.add(section_name)
        
        # Determine section boundaries
        for i, (start_pos, section_name, variation) in enumerate(unique_headers):
            # Find end position (start of next section or end of document)
            if i + 1 < len(unique_headers):
                end_pos = unique_headers[i + 1][0]
            else:
                end_pos = len(normalized_text)
            
            section_positions[section_name] = (start_pos, end_pos)
        
        # Special handling for FORWARDING LETTER in customer copy
        if doc_type == "customer" and "FORWARDING LETTER" in section_positions:
            forwarding_start, _ = section_positions["FORWARDING LETTER"]
            
            # Find where PREAMBLE starts to end FORWARDING LETTER
            if "PREAMBLE" in section_positions:
                preamble_start, preamble_end = section_positions["PREAMBLE"]
                section_positions["FORWARDING LETTER"] = (forwarding_start, preamble_start)
            else:
                # If no PREAMBLE found, look for other indicators
                # Search for common forwarding letter end patterns
                forwarding_text = normalized_text[forwarding_start:]
                end_patterns = [
                    "DISCLAIMER", "IN CASE OF DISPUTE", "ENGLISH VERSION",
                    "REGARDS", "YOURS FAITHFULLY", "SINCERELY"
                ]
                
                min_end_pos = len(normalized_text)
                for pattern in end_patterns:
                    pattern_pos = forwarding_text.find(pattern)
                    if pattern_pos != -1:
                        # Find end of this pattern (next line break or end)
                        pattern_end = forwarding_text.find('\n', pattern_pos)
                        if pattern_end == -1:
                            pattern_end = len(forwarding_text)
                        actual_end_pos = forwarding_start + pattern_end
                        min_end_pos = min(min_end_pos, actual_end_pos)
                
                if min_end_pos < len(normalized_text):
                    section_positions["FORWARDING LETTER"] = (forwarding_start, min_end_pos)
        
        return section_positions

    def extract_section_content(self, text: str, section_name: str, start_pos: int, end_pos: int) -> str:
        """Extract the actual content of a section given its boundaries"""
        if start_pos >= len(text) or end_pos <= start_pos:
            return "NOT FOUND"
        
        # Extract the section content
        section_content = text[start_pos:end_pos].strip()
        
        if not section_content:
            return "NOT FOUND"
        
        # Clean up the content
        # Remove excessive whitespace
        section_content = re.sub(r'\n\s*\n', '\n\n', section_content)
        section_content = re.sub(r' +', ' ', section_content)
        
        return section_content

    def extract_sections_with_python(self, document_text: str, doc_name: str, doc_type: str = "filed") -> Dict[str, str]:
        """
        Extract sections using Python-based header identification
        doc_type: "filed" or "customer"
        """
        try:
            st.info(f"Extracting sections from {doc_name} using Python header identification...")
            
            # Find section boundaries
            section_boundaries = self.find_section_boundaries(document_text, doc_type)
            
            # Extract content for each target section
            extracted_sections = {}
            
            for section_name in self.section_headers:
                if section_name in section_boundaries:
                    start_pos, end_pos = section_boundaries[section_name]
                    content = self.extract_section_content(document_text, section_name, start_pos, end_pos)
                    extracted_sections[section_name] = content
                    
                    st.write(f"✅ Found {section_name}: {len(content)} characters")
                else:
                    extracted_sections[section_name] = "NOT FOUND"
                    st.write(f"❌ {section_name}: NOT FOUND")
            
            return extracted_sections
            
        except Exception as e:
            logger.error(f"Error in Python section extraction for {doc_name}: {str(e)}")
            st.error(f"Error extracting sections from {doc_name}: {str(e)}")
            
            # Return default structure
            return {section: "NOT FOUND" for section in self.section_headers}

    def compare_documents_with_llm(self, doc1_sections: Dict[str, str], doc2_sections: Dict[str, str], 
                               doc1_name: str, doc2_name: str, sample_number: str) -> List[Dict]:
        """Compare documents using LLM analysis"""
        comparison_results = []

        for section in self.section_headers:
            doc1_content = doc1_sections.get(section, "NOT FOUND")
            doc2_content = doc2_sections.get(section, "NOT FOUND")

            total_content_size = len(doc1_content) + len(doc2_content)
            
            if total_content_size > 8000:
                # Truncate content for comparison but keep full content for display
                doc1_truncated = doc1_content[:3000] if doc1_content != "NOT FOUND" else "NOT FOUND"
                doc2_truncated = doc2_content[:3000] if doc2_content != "NOT FOUND" else "NOT FOUND"
                
                st.warning(f"Section {section} content truncated for comparison due to size")
                
                comparison_result = self._compare_section_content(
                    doc1_truncated, doc2_truncated, section, sample_number, 
                    doc1_content, doc2_content
                )
            else:
                comparison_result = self._compare_section_content(
                    doc1_content, doc2_content, section, sample_number
                )
            
            comparison_results.append(comparison_result)

        return comparison_results
    
    def _compare_section_content(self, doc1_content: str, doc2_content: str, section: str, 
                               sample_number: str, full_doc1_content: str = None, 
                               full_doc2_content: str = None) -> Dict:
        """Compare content of a specific section using LLM"""
        
        display_doc1 = full_doc1_content if full_doc1_content is not None else doc1_content
        display_doc2 = full_doc2_content if full_doc2_content is not None else doc2_content

        system_prompt = f"""You are a document comparison expert. Your goal is to analyze the following two versions of the same section and do the following:

Step 1: Understand each section separately.

Step 2: Identify all *meaningful content differences* between them, focusing only on:
- Contextual changes
- Textual changes
- Modifications, additions and deletions
- Ignore numbering or serialization differences

IGNORE differences in formatting, punctuation, line breaks, numbering or case changes.

Step 3: Present a structured, **point-wise list** of the meaningful differences, e.g.:
1. Date changed from 'X' in Document 1 to 'Y' in Document 2.
2. The clause about <topic> is present in Document 2 but missing in Document 1.
3. Name changed from 'Mr. X' to 'Mr. Y'.

Section Name: {section}

Compare the two documents as:
- Filed Copy = Document 1
- Customer Copy = Document 2

Filed Copy:
{doc1_content}

Customer Copy:
{doc2_content}

Respond only with the final response after understanding and following the above steps. If no meaningful content differences are found, clearly respond: "NO_CONTENT_DIFFERENCE".
"""

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
        """Get display name for section"""
        section_mapping = {
            "FORWARDING LETTER": "Forwarding letter",
            "PREAMBLE": "Preamble", 
            "SCHEDULE": "Schedule",
            "DEFINITIONS & ABBREVIATIONS": "Definitions and Abbreviations"
        }
        return section_mapping.get(section, section.title())
    
    def _create_section_result(self, response_content: str, section: str, doc1_content: str, 
                             doc2_content: str, sample_number: str) -> Dict:
        """Create structured result for a section comparison"""
        
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
        """Prepare content for display in results"""
        
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
        """Parse and clean the difference description from LLM response"""

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
        """Create Excel report from comparison results"""
        
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
        
        output.seek(0)
        return output.getvalue()

def main():
    st.set_page_config(
        page_title="Document Comparer with Python Section Extraction",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Comparer with Python Section Extraction")
    st.markdown("Upload two PDF documents to compare specific sections using **Python-based header identification**")
    
    # Sidebar for Azure OpenAI configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        azure_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            placeholder="https://your-resource.openai.azure.com/",
            type="password"
        )
        
        api_key = st.text_input(
            "API Key",
            placeholder="Enter your Azure OpenAI API key",
            type="password"
        )
        
        api_version = st.text_input(
            "API Version",
            value="2024-02-01",
            placeholder="2024-02-01"
        )
        
        deployment_name = st.text_input(
            "Deployment Name",
            placeholder="gpt-4",
            value="gpt-4"
        )
        
        st.markdown("---")
        st.markdown("**Target Sections:**")
        st.markdown("• Forwarding Letter")
        st.markdown("• Preamble")
        st.markdown("• Schedule")
        st.markdown("• Definitions & Abbreviations")
        
        st.markdown("---")
        st.markdown("**New Features:**")
        st.markdown("✅ Python-based section extraction")
        st.markdown("✅ Header pattern matching")
        st.markdown("✅ Special handling for customer copy")
        st.markdown("✅ Flexible header variations")
        st.markdown("✅ LLM-based content comparison")
    
    # Main interface
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
    if st.button("🔍 Analyze with Python Section Extraction", type="primary"):
        if not all([azure_endpoint, api_key, deployment_name]):
            st.error("❌ Please provide all Azure OpenAI configuration details")
            return
        
        if not doc1_file or not doc2_file:
            st.error("❌ Please upload both PDF documents")
            return
        
        try:
            # Initialize comparer
            comparer = DocumentComparer(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                deployment_name=deployment_name
            )
            
            with st.spinner("🔄 Processing documents..."):
                # Extract text from PDFs
                st.info("📖 Extracting text from documents...")
                doc1_text = comparer.extract_text_from_pdf(doc1_file)
                doc2_text = comparer.extract_text_from_pdf(doc2_file)
                
                if not doc1_text or not doc2_text:
                    st.error("❌ Failed to extract text from one or both documents")
                    return
                
                # Extract sample number from document 2 (Customer Copy)
                sample_number = comparer.extract_sample_number_from_filename(doc2_file.name)
                
                # Extract sections using Python-based method
                st.info("🐍 Extracting sections using Python header identification...")
                
                with st.expander("Filed Copy Section Extraction"):
                    doc1_sections = comparer.extract_sections_with_python(doc1_text, doc1_file.name, "filed")
                
                with st.expander("Customer Copy Section Extraction"):
                    doc2_sections = comparer.extract_sections_with_python(doc2_text, doc2_file.name, "customer")
                
                # Compare documents using LLM
                st.info("🔍 Comparing extracted sections with AI analysis...")
                comparison_results = comparer.compare_documents_with_llm(
                    doc1_sections, doc2_sections, doc1_file.name, doc2_file.name, sample_number
                )
                
                # Create Excel report
                st.info("📊 Generating Excel report...")
                excel_data = comparer.create_excel_report(
                    comparison_results, doc1_file.name, doc2_file.name
                )
            
            st.success("✅ Document analysis completed successfully with Python section extraction!")
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            # Summary metrics
            sections_with_differences = len([r for r in comparison_results if 'Mismatch' in r.get('Observation - Category', '')])
            sections_without_differences = len(comparison_results) - sections_with_differences
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Sections Analyzed", len(comparer.section_headers))
            
            with col2:
                st.metric("Sections with Differences", sections_with_differences)
            
            with col3:
                st.metric("Sections without Differences", sections_without_differences)
            
            with col4:
                st.metric("Sample Number", sample_number)
            
            # Show extracted sections
            st.subheader("📋 Extracted Sections Preview")
            
            for section in comparer.section_headers:
                with st.expander(f"{section} - Content Preview"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Filed Copy:**")
                        filed_content = doc1_sections.get(section, "NOT FOUND")
                        if filed_content != "NOT FOUND":
                            st.text_area("", filed_content[:500] + ("..." if len(filed_content) > 500 else ""), height=150, disabled=True)
                        else:
                            st.warning("NOT FOUND")
                    
                    with col2:
                        st.write("**Customer Copy:**")
                        customer_content = doc2_sections.get(section, "NOT FOUND")
                        if customer_content != "NOT FOUND":
                            st.text_area("", customer_content[:500] + ("..." if len(customer_content) > 500 else ""), height=150, disabled=True, key=f"customer_{section}")
                        else:
                            st.warning("NOT FOUND")
            
            # Detailed results table
            st.subheader("📋 Comparison Results")
            df_display = pd.DataFrame(comparison_results)
            
            # Create a display version without the full content
            df_display_short = df_display.copy()
            df_display_short['Content (Preview)'] = df_display_short['Content'].apply(
                lambda x: x[:200] + "..." if len(str(x)) > 200 else x
            )
            
            display_columns = ['Samples affected', 'Observation - Category', 'Page', 'Sub-category of Observation', 'Content (Preview)']
            st.dataframe(df_display_short[display_columns], use_container_width=True)
            
            # Download Excel report
            st.subheader("📥 Download Report")
            st.download_button(
                label="📊 Download Complete Analysis Report",
                data=excel_data,
                file_name=f"python_extraction_analysis_{sample_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
