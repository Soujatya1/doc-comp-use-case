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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentComparer:
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str, deployment_name: str):
        """Initialize the document comparer with Azure OpenAI configuration."""
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
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_sample_number(self, document_text: str) -> str:
        """Extract sample number from document text."""
        # Common patterns for sample numbers
        patterns = [
            r'Sample\s*[#:]?\s*(\d+)',
            r'Sample\s+No\.?\s*(\d+)',
            r'Sample\s+Number\s*[:]?\s*(\d+)',
            r'Doc\s*[#:]?\s*(\d+)',
            r'Document\s*[#:]?\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no pattern found, return default
        return "001"
    
    def filter_sections_with_llm(self, document_text: str, doc_name: str) -> Dict[str, str]:
        """Use LLM to filter and extract specific sections from document."""
        
        system_prompt = """You are an expert document analyzer. Your task is to extract specific sections from legal/business documents.

Target sections to extract:
1. FORWARDING LETTER: Starts with "Sub: issuance of the policy under application for the life insurance policy dated" and ends with "Disclaimer: In case of dispute, English version of Policy bond shall be final and binding."
2. PREAMBLE: Starts with "The Company has received a Proposal Form, declaration and the first Regular Premium from the Policyholder / Life Assured as named in this Schedule." and ends with "incorporated herein and forms the basis of this Policy."
3. SCHEDULE
4. DEFINITIONS & ABBREVIATIONS

Instructions:
- Identify and extract ONLY the content from these sections
- Include section headers in the extracted content
- If a section is not found, return "NOT FOUND" for that section
- Maintain the original formatting and structure
- Be precise and extract complete sections without truncation

Return the result as a JSON object with section names as keys and extracted content as values."""

        user_prompt = f"""Please analyze the following document and extract the target sections:

Document Name: {doc_name}

Document Content:
{document_text}

Extract the sections and return as JSON format:
{{
    "FORWARDING LETTER": "extracted content or NOT FOUND",
    "PREAMBLE": "extracted content or NOT FOUND", 
    "SCHEDULE": "extracted content or NOT FOUND",
    "DEFINITIONS & ABBREVIATIONS": "extracted content or NOT FOUND"
}}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                sections_dict = json.loads(json_match.group())
                return sections_dict
            else:
                logger.warning("Could not parse JSON from LLM response")
                return self._fallback_section_extraction(document_text)
                
        except Exception as e:
            logger.error(f"Error in LLM section filtering: {str(e)}")
            return self._fallback_section_extraction(document_text)
    
    def _fallback_section_extraction(self, text: str) -> Dict[str, str]:
        """Fallback method for section extraction using regex."""
        sections = {}
        text_upper = text.upper()
        
        for section in self.target_sections:
            section_upper = section.upper()
            pattern = rf"({re.escape(section_upper)}.*?)(?=\n[A-Z\d]+\.|$)"
            match = re.search(pattern, text_upper, re.DOTALL)
            
            if match:
                sections[section] = match.group(1).strip()
            else:
                sections[section] = "NOT FOUND"
        
        return sections
    
    def compare_documents_with_llm(self, doc1_sections: Dict[str, str], doc2_sections: Dict[str, str], 
                                 doc1_name: str, doc2_name: str, sample_number: str) -> List[Dict]:
        """Use LLM to intelligently compare document sections and create results for ALL sections."""
        
        system_prompt = """You are a document comparison expert. Your goal is to analyze the following two versions of the same section and do the following:

Step 1: Understand and summarize the core content of each section separately.

Step 2: Identify all *meaningful content differences* between them, focusing only on:
- Different names, addresses, contact details
- Changes in numbers, dates, percentages
- Missing or additional sentences, clauses, or legal terms
- Modified terms, conditions, or contractual language
- Any altered product names, policy identifiers, or descriptions

IGNORE differences in formatting, punctuation, line breaks, or case changes.

Step 3: Present a structured, **point-wise list** of the meaningful differences, e.g.:
1. Date changed from 'X' in Document 1 to 'Y' in Document 2.
2. The clause about <topic> is present in Document 2 but missing in Document 1.
3. Name changed from 'Mr. X' to 'Mr. Y'.

Section Name: {section}

Document 1 ({doc1_name}) - Filed Copy:
{doc1_content}

Document 2 ({doc2_name}) - Customer Copy:
{doc2_content}

Respond only with your comparison as per steps 1 to 3. If no meaningful content differences are found, clearly respond: "NO_CONTENT_DIFFERENCE"."""

            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = self.llm.invoke(messages)
                
                # Create result for this section (regardless of differences)
                comparison_result = self._create_section_result(
                    response.content, section, doc1_content, doc2_content, sample_number
                )
                comparison_results.append(comparison_result)
                
            except Exception as e:
                logger.error(f"Error comparing section {section}: {str(e)}")
                comparison_results.append({
                    'Samples affected': sample_number,
                    'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
                    'Page': self._get_page_name(section),
                    'Sub-category of Observation': f'Error during comparison: {str(e)}',
                    'Content': f"Filed Copy: {doc1_content[:500]}{'...' if len(doc1_content) > 500 else ''}\n\nCustomer Copy: {doc2_content[:500]}{'...' if len(doc2_content) > 500 else ''}"
                })
        
        return comparison_results
    
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
        """Create result for each section, showing content regardless of differences."""
        
        # Check if there are no meaningful content differences
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
        
        # Also check for responses that only mention formatting/structural differences
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
        
        # Prepare content for display
        content_display = self._prepare_content_display(doc1_content, doc2_content, section)
        
        # Handle missing sections
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
        
        # Determine observation category and sub-category
        if has_differences:
            # Parse the LLM response for meaningful differences
            sub_category = self._parse_difference_description(response_content)
            observation_category = 'Mismatch of content between Filed Copy and customer copy'
        else:
            # No meaningful content differences
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
        """Prepare content display for the Content column."""
        
        # If both contents are identical, show once
        if doc1_content == doc2_content and doc1_content != "NOT FOUND":
            content = doc1_content
            # Truncate if too long for Excel display
            if len(content) > 1000:
                content = content[:1000] + "... [Content truncated for display]"
            return f"[IDENTICAL CONTENT]\n{content}"
        
        # If contents are different, show both
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
    
    def _parse_difference_description(self, response_content: str) -> str:
        """Parse LLM response to extract difference description."""
        
        # Clean up the LLM response to use as sub-category
        sub_category = response_content.strip()
        
        # Remove common prefixes that might be added by LLM (case-insensitive)
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
        
        # Clean up common suffixes and extra whitespace
        sub_category = re.sub(r'\s+', ' ', sub_category)
        sub_category = sub_category.strip('. \n\r\t')
        
        # Remove bullet points and numbering from the beginning
        sub_category = re.sub(r'^[-•*]\s*', '', sub_category)
        sub_category = re.sub(r'^\d+\.\s*', '', sub_category)
        
        # If response contains multiple lines, take the first meaningful line
        lines = [line.strip() for line in sub_category.split('\n') if line.strip()]
        if lines:
            for line in lines:
                if len(line) > 10:
                    sub_category = line
                    break
            else:
                sub_category = lines[0]
        
        # Truncate if too long
        if len(sub_category) > 500:
            truncate_at = 497
            last_sentence_end = max(
                sub_category.rfind('.', 0, truncate_at),
                sub_category.rfind('!', 0, truncate_at),
                sub_category.rfind('?', 0, truncate_at)
            )
            if last_sentence_end > 200:
                sub_category = sub_category[:last_sentence_end + 1]
            else:
                sub_category = sub_category[:497] + "..."
        
        # Ensure it starts with uppercase
        if sub_category and len(sub_category) > 0:
            sub_category = sub_category[0].upper() + sub_category[1:]
        
        # Default if too generic or short
        if len(sub_category) < 10:
            sub_category = f"Meaningful content differences identified in section"
        
        return sub_category
    
    def create_excel_report(self, comparison_results: List[Dict], doc1_name: str, doc2_name: str) -> bytes:
        """Create Excel report from comparison results with proper formatting including Content column."""
        
        # Create DataFrame
        df = pd.DataFrame(comparison_results)
        
        # Reorder columns to include the new Content column
        if not df.empty:
            df = df[['Samples affected', 'Observation - Category', 'Page', 'Sub-category of Observation', 'Content']]
        
        # Create Excel file in memory
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main comparison sheet
            df.to_excel(writer, sheet_name='Document_Comparison', index=False)
            
            # Format the main sheet for better readability
            if not df.empty:
                workbook = writer.book
                worksheet = writer.sheets['Document_Comparison']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value:
                                # Handle different columns specially
                                if column_letter == 'D':  # Sub-category of Observation column
                                    max_length = max(max_length, min(len(str(cell.value)), 80))
                                elif column_letter == 'E':  # Content column
                                    max_length = max(max_length, min(len(str(cell.value)), 120))
                                else:
                                    max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    
                    # Set column width with reasonable limits
                    if column_letter == 'D':  # Sub-category of Observation column
                        adjusted_width = min(max_length + 2, 80)
                    elif column_letter == 'E':  # Content column
                        adjusted_width = min(max_length + 2, 120)  # Wider for content
                    else:
                        adjusted_width = min(max_length + 2, 30)
                    
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Enable text wrapping for all cells
                from openpyxl.styles import Alignment
                for row in worksheet.iter_rows():
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            # Summary sheet
            sections_with_differences = len([r for r in comparison_results if 'Mismatch' in r.get('Observation - Category', '')])
            sections_without_differences = len(comparison_results) - sections_with_differences
            
            summary_data = {
                'Metric': [
                    'Total Sections Analyzed',
                    'Sections with Content Differences',
                    'Sections without Content Differences',
                    'Document 1 Name (Filed Copy)',
                    'Document 2 Name (Customer Copy)',
                    'Comparison Date',
                    'Analysis Type'
                ],
                'Value': [
                    len(self.target_sections),
                    sections_with_differences,
                    sections_without_differences,
                    doc1_name,
                    doc2_name,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Complete Section Analysis with Content Display'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Format summary sheet
            summary_worksheet = writer.sheets['Summary']
            for column in summary_worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if cell.value:
                            max_length = max(max_length, len(str(cell.value)))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 80)
                summary_worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Enable text wrapping for summary sheet
            for row in summary_worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
        
        output.seek(0)
        return output.getvalue()

def main():
    st.set_page_config(
        page_title="Enhanced Document Comparer with Content Display",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Enhanced Document Comparer with Content Display")
    st.markdown("Upload two PDF documents to compare specific sections and view **all extracted content** with detailed analysis")
    
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
        st.markdown("**Analysis Features:**")
        st.markdown("✅ All sections shown")
        st.markdown("✅ Content differences highlighted")
        st.markdown("✅ Complete content display")
        st.markdown("✅ Structured Excel output")
    
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
    if st.button("🔍 Analyze All Sections with Content", type="primary"):
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
                sample_number = comparer.extract_sample_number(doc2_text)
                
                # Filter sections using LLM
                st.info("🤖 Filtering sections using AI...")
                doc1_sections = comparer.filter_sections_with_llm(doc1_text, doc1_file.name)
                doc2_sections = comparer.filter_sections_with_llm(doc2_text, doc2_file.name)
                
                # Compare documents using LLM (now includes all sections)
                st.info("🔍 Analyzing all sections with content...")
                comparison_results = comparer.compare_documents_with_llm(
                    doc1_sections, doc2_sections, doc1_file.name, doc2_file.name, sample_number
                )
                
                # Create Excel report
                st.info("📊 Generating comprehensive Excel report...")
                excel_data = comparer.create_excel_report(
                    comparison_results, doc1_file.name, doc2_file.name
                )
            
            st.success("✅ Complete document analysis with content display completed successfully!")
            
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
                file_name=f"complete_analysis_{sample_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
