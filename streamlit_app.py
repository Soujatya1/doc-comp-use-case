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
        """Use LLM to intelligently compare document sections and find differences."""
        
        system_prompt = """You are an expert document comparison analyst. Your task is to intelligently compare corresponding sections from two documents and identify meaningful differences.

Instructions:
- Compare each section between the two documents
- Identify content differences, structural changes, additions, deletions, and modifications
- Focus on meaningful differences, not minor formatting variations
- If sections are identical, note as "NO_DIFFERENCE"
- If a section is missing in one document, note as "MISSING_IN_DOC1" or "MISSING_IN_DOC2"
- Provide a concise but descriptive summary of what specifically changed, added, or removed
- Be specific about the nature of differences (e.g., "Date changed from X to Y", "Clause added about Z", "Amount modified", etc.)

For each difference found, provide a brief but informative description of the specific changes."""

        comparison_results = []
        
        for section in self.target_sections:
            section_key = section.upper()
            doc1_content = doc1_sections.get(section_key, "NOT FOUND")
            doc2_content = doc2_sections.get(section_key, "NOT FOUND")
            
            user_prompt = f"""Compare the following section between two documents:

Section: {section}

Document 1 ({doc1_name}) - Filed Copy:
{doc1_content}

Document 2 ({doc2_name}) - Customer Copy:
{doc2_content}

Analyze and provide a specific description of differences found. Focus on what exactly changed, was added, or was removed. If no meaningful differences exist, respond with "NO_DIFFERENCE"."""

            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = self.llm.invoke(messages)
                
                # Parse the comparison result
                comparison_result = self._parse_comparison_response(
                    response.content, section, doc1_content, doc2_content, sample_number
                )
                if comparison_result:
                    comparison_results.append(comparison_result)
                
            except Exception as e:
                logger.error(f"Error comparing section {section}: {str(e)}")
                comparison_results.append({
                    'Samples affected': sample_number,
                    'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
                    'Page': self._get_page_name(section),
                    'Sub-category of Observation': f'Error during comparison: {str(e)}'
                })
        
        return comparison_results
    
    def _get_page_name(self, section: str) -> str:
        """Map sections to appropriate page names as shown in the image."""
        section_mapping = {
            "FORWARDING LETTER": "Forwarding letter",
            "PREAMBLE": "Preamble", 
            "SCHEDULE": "Schedule",
            "DEFINITIONS & ABBREVIATIONS": "Definitions and Abbreviations"
        }
        return section_mapping.get(section, section.title())
    
    def _parse_comparison_response(self, response_content: str, section: str, doc1_content: str, 
                             doc2_content: str, sample_number: str) -> Dict:
        """Parse LLM comparison response into structured format."""
    
        # Check if there are no differences
        if ("NO_DIFFERENCE" in response_content.upper() or 
            "IDENTICAL" in response_content.upper() or
            "NO MEANINGFUL DIFFERENCES" in response_content.upper()):
            return None  # Return None for no differences
        
        # Check for missing sections
        if doc1_content == "NOT FOUND" and doc2_content == "NOT FOUND":
            return {
                'Samples affected': sample_number,
                'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
                'Page': self._get_page_name(section),
                'Sub-category of Observation': 'Section not found in both documents'
            }
        elif doc1_content == "NOT FOUND":
            return {
                'Samples affected': sample_number,
                'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
                'Page': self._get_page_name(section),
                'Sub-category of Observation': 'Section missing in Filed Copy'
            }
        elif doc2_content == "NOT FOUND":
            return {
                'Samples affected': sample_number,
                'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
                'Page': self._get_page_name(section),
                'Sub-category of Observation': 'Section missing in Customer Copy'
            }
        
        # Clean up the LLM response to use as sub-category
        sub_category = response_content.strip()
        
        # Remove common prefixes that might be added by LLM
        prefixes_to_remove = [
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
        
        # Limit length and ensure it's informative
        if len(sub_category) > 200:
            sub_category = sub_category[:197] + "..."
        
        # If the response is too generic, provide a more specific description
        if len(sub_category) < 10 or sub_category.lower() in ["differences found", "content differs", "changes detected"]:
            sub_category = f"Content differences identified in {self._get_page_name(section)} section"
        
        return {
            'Samples affected': sample_number,
            'Observation - Category': 'Mismatch of content between Filed Copy and customer copy',
            'Page': self._get_page_name(section),
            'Sub-category of Observation': sub_category
        }
    
    def create_excel_report(self, comparison_results: List[Dict], doc1_name: str, doc2_name: str) -> bytes:
        """Create Excel report from comparison results."""
        
        # Create DataFrame
        df = pd.DataFrame(comparison_results)
        
        # Reorder columns to match the required format
        if not df.empty:
            df = df[['Samples affected', 'Observation - Category', 'Page', 'Sub-category of Observation']]
        
        # Create Excel file in memory
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main comparison sheet
            df.to_excel(writer, sheet_name='Document_Comparison', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Sections Compared',
                    'Sections with Differences',
                    'Document 1 Name (Filed Copy)',
                    'Document 2 Name (Customer Copy)',
                    'Comparison Date'
                ],
                'Value': [
                    len(self.target_sections),
                    len(comparison_results),
                    doc1_name,
                    doc2_name,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return output.getvalue()

def main():
    st.set_page_config(
        page_title="Intelligent Document Comparer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Intelligent Document Comparer")
    st.markdown("Upload two PDF documents to compare specific sections using AI-powered analysis")
    
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
    if st.button("🔍 Compare Documents", type="primary"):
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
                
                # Compare documents using LLM
                st.info("🔍 Comparing documents using AI...")
                comparison_results = comparer.compare_documents_with_llm(
                    doc1_sections, doc2_sections, doc1_file.name, doc2_file.name, sample_number
                )
                
                # Create Excel report
                st.info("📊 Generating Excel report...")
                excel_data = comparer.create_excel_report(
                    comparison_results, doc1_file.name, doc2_file.name
                )
            
            st.success("✅ Document comparison completed successfully!")
            
            # Display results
            st.subheader("📊 Comparison Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Sections Compared", len(comparer.target_sections))
            
            with col2:
                st.metric("Sections with Differences", len(comparison_results))
            
            with col3:
                st.metric("Sample Number", sample_number)
            
            # Detailed results table
            if comparison_results:
                st.subheader("📋 Detailed Comparison")
                df_display = pd.DataFrame(comparison_results)
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info("✅ No differences found between the documents!")
            
            # Download Excel report
            st.subheader("📥 Download Report")
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"document_comparison_{sample_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
