import streamlit as st
import pandas as pd
import PyPDF2
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
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str, deployment_name: str):
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
            "DEFINITIONS & ABBREVIATIONS"
        ]
        
        # Define section headers for header identification
        self.SECTION_HEADERS = [
            "FORWARDING LETTER",
            "PREAMBLE", 
            "SCHEDULE",
            "DEFINITIONS & ABBREVIATIONS",
            "DEFINITIONS",
            "ABBREVIATIONS"
        ]
        
        self.max_input_chars = 12000
        self.chunk_size = 8000

    def extract_text_from_pdf(self, pdf_file) -> str:
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

    def extract_sections(self, text):
        """Extract sections using header identification method"""
        sections = {}
        current_section = None
        
        for line in text.splitlines():
            line_stripped = line.strip().upper()
            if line_stripped in self.SECTION_HEADERS:
                current_section = line_stripped
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)
        
        # Convert lists to strings and normalize section names
        result_sections = {}
        for k, v in sections.items():
            content = "\n".join(v).strip()
            # Normalize section names to match target_sections
            if k in ["DEFINITIONS", "ABBREVIATIONS"]:
                result_sections["DEFINITIONS & ABBREVIATIONS"] = content
            else:
                result_sections[k] = content
                
        return result_sections

    def extract_forwarding_letter_section_filed(self, text):
        """Special case: FORWARDING LETTER extraction from Filed Copy"""
        lines = text.splitlines()
        try:
            start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "FORWARDING LETTER")
            end_idx = next(i for i, line in enumerate(lines[start_idx+1:], start=start_idx+1)
                          if line.strip().upper() in self.SECTION_HEADERS and line.strip().upper() != "FORWARDING LETTER")
            return "\n".join(lines[start_idx:end_idx]).strip()
        except StopIteration:
            return "FORWARDING LETTER section not found in Filed Copy."

    def extract_forwarding_letter_section_customer(self, text):
        """Special case: FORWARDING LETTER extraction from Customer Copy"""
        lines = text.splitlines()
        try:
            end_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "PREAMBLE")
            return "\n".join(lines[:end_idx]).strip()
        except StopIteration:
            return "PREAMBLE section not found in Customer Copy; cannot extract FORWARDING LETTER."

    def extract_sections_with_python(self, document_text: str, doc_name: str, is_filed_copy: bool = True) -> Dict[str, str]:
        """Extract sections using Python header identification methods"""
        try:
            st.info(f"Extracting sections from {doc_name} using header identification...")
            
            # Extract general sections
            sections = self.extract_sections(document_text)
            
            # Handle FORWARDING LETTER specially based on document type
            if is_filed_copy:
                forwarding_letter = self.extract_forwarding_letter_section_filed(document_text)
            else:
                forwarding_letter = self.extract_forwarding_letter_section_customer(document_text)
            
            sections["FORWARDING LETTER"] = forwarding_letter
            
            # Ensure all target sections are present
            final_sections = {}
            for section in self.target_sections:
                if section in sections and sections[section]:
                    final_sections[section] = sections[section]
                else:
                    final_sections[section] = "NOT FOUND"
            
            st.success(f"Successfully extracted sections from {doc_name}")
            logger.info(f"Extracted sections from {doc_name}: {list(final_sections.keys())}")
            
            return final_sections
            
        except Exception as e:
            logger.error(f"Error extracting sections from {doc_name}: {str(e)}")
            # Return default structure if extraction fails
            return {section: "NOT FOUND" for section in self.target_sections}

    def compare_documents_with_llm(self, doc1_sections: Dict[str, str], doc2_sections: Dict[str, str], 
                               doc1_name: str, doc2_name: str, sample_number: str) -> List[Dict]:
    
        comparison_results = []

        for section in self.target_sections:
            doc1_content = doc1_sections.get(section, "NOT FOUND")
            doc2_content = doc2_sections.get(section, "NOT FOUND")

            total_content_size = len(doc1_content) + len(doc2_content)
            
            if total_content_size > 8000:  # If combined content is too large
                # Truncate content for comparison but keep full content for display
                doc1_truncated = doc1_content[:3000] if doc1_content != "NOT FOUND" else "NOT FOUND"
                doc2_truncated = doc2_content[:3000] if doc2_content != "NOT FOUND" else "NOT FOUND"
                
                st.warning(f"Section {section} content truncated for comparison due to size")
                
                comparison_result = self._compare_section_content(
                    doc1_truncated, doc2_truncated, section, sample_number, 
                    doc1_content, doc2_content  # Pass full content for display
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
            # Parse the LLM response for meaningful differences
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

        # Default if too generic or short
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
                if total_rows > 1:  # Only merge if there's more than 1 row
                    start_row = 2  # Row 2 (after header)
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
                
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value:
                                # Handle different columns specially
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
        page_title="Enhanced Document Comparer with Header-Based Extraction",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Enhanced Document Comparer with Header-Based Extraction")
    st.markdown("Upload two PDF documents to compare specific sections using **Python header identification** methods")
    
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
        st.markdown("**Extraction Method:**")
        st.markdown("🐍 **Python Header Identification**")
        st.markdown("✅ Fast header-based extraction")
        st.markdown("✅ Special handling for Forwarding Letter")
        st.markdown("✅ AI-powered content comparison")
        st.markdown("✅ Complete content display")
    
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
    if st.button("🔍 Analyze Sections with Header Identification", type="primary"):
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
                
                # Extract sections using Python header identification
                st.info("🐍 Extracting sections using Python header identification...")
                doc1_sections = comparer.extract_sections_with_python(doc1_text, doc1_file.name, is_filed_copy=True)
                logger.info("📄 Filed Copy Extracted Sections:\n" + json.dumps(doc1_sections, indent=2))
                doc2_sections = comparer.extract_sections_with_python(doc2_text, doc2_file.name, is_filed_copy=False)
                logger.info("📄 Customer Copy Extracted Sections:\n" + json.dumps(doc2_sections, indent=2))
                
                # Compare documents using LLM (content comparison only)
                st.info("🔍 Analyzing sections with AI comparison...")
                comparison_results = comparer.compare_documents_with_llm(
                    doc1_sections, doc2_sections, doc1_file.name, doc2_file.name, sample_number
                )
                
                # Create Excel report
                st.info("📊 Generating comprehensive Excel report...")
                excel_data = comparer.create_excel_report(
                    comparison_results, doc1_file.name, doc2_file.name
                )
            
            st.success("✅ Document analysis with header-based extraction completed successfully!")
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            # Summary metrics
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
            
            # Detailed results table
            st.subheader("📋 Complete Section Analysis")
            df_display = pd.DataFrame(comparison_results)
            
            # Create a display version without the full content for better viewing
            df_display_short = df_display.copy()
            df_display_short['Content (Preview)'] = df_display_short['Content'].apply(
                lambda x: x[:200] + "..." if len(str(x)) > 200 else x
            )
            
            # Show table without full content column for better display
            display_columns = ['Samples affected', 'Observation - Category', 'Page', 'Sub-category of Observation', 'Content (Preview)']
            st.dataframe(df_display_short[display_columns], use_container_width=True)
            
            st.info("ℹ️ **Note**: Complete content is available in the downloadable Excel report. Preview shows first 200 characters.")
            
            # Download Excel report
            st.subheader("📥 Download Complete Report")
            st.download_button(
                label="📊 Download Complete Analysis with Content",
                data=excel_data,
                file_name=f"header_based_analysis_{sample_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officeedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
