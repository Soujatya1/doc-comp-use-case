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
    
    def truncate_content_smart(self, content: str, first_words: int = 500, last_words: int = 100) -> str:
        """
        Truncate content to show first N words and last M words with separator.
        
        Args:
            content: The text content to truncate
            first_words: Number of words to show from the beginning
            last_words: Number of words to show from the end
            
        Returns:
            Truncated content string
        """
        if content in ["NOT FOUND", ""]:
            return content
            
        words = content.split()
        total_words = len(words)
        
        # If content is short enough, return as is
        if total_words <= (first_words + last_words):
            return content
        
        # Get first N words
        first_part = " ".join(words[:first_words])
        
        # Get last M words
        last_part = " ".join(words[-last_words:])
        
        # Combine with separator
        truncated = f"{first_part}\n\n... [CONTENT TRUNCATED - {total_words - first_words - last_words} words omitted] ...\n\n{last_part}"
        
        return truncated
    
    def filter_sections_with_llm(self, document_text: str, doc_name: str) -> Dict[str, str]:
        """Use LLM to filter and extract specific sections from document."""
        
        system_prompt = """You are an expert document analyzer. Your task is to extract specific sections from legal/business documents.

Target sections to extract:
1. FORWARDING LETTER
2. PREAMBLE
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
                                 doc1_name: str, doc2_name: str) -> List[Dict]:
        """Use LLM to intelligently compare document sections and find differences."""
        
        system_prompt = """You are an expert document comparison analyst. Your task is to intelligently compare corresponding sections from two documents and identify meaningful differences.

Instructions:
- Compare each section between the two documents
- Identify content differences, structural changes, additions, deletions, and modifications
- Focus on meaningful differences, not minor formatting variations
- Categorize differences as: ADDED, REMOVED, MODIFIED, STRUCTURAL_CHANGE
- Provide specific details about what changed
- If sections are identical, note as "NO_DIFFERENCE"
- If a section is missing in one document, note as "MISSING_IN_DOC1" or "MISSING_IN_DOC2"

For each difference found, provide:
1. Section name
2. Difference type (ADDED/REMOVED/MODIFIED/STRUCTURAL_CHANGE/NO_DIFFERENCE/MISSING)
3. Description of the difference
4. Content from Document 1 (if applicable)
5. Content from Document 2 (if applicable)
6. Impact level (HIGH/MEDIUM/LOW)"""

        comparison_results = []
        
        for section in self.target_sections:
            section_key = section.upper()
            doc1_content = doc1_sections.get(section_key, "NOT FOUND")
            doc2_content = doc2_sections.get(section_key, "NOT FOUND")
            
            user_prompt = f"""Compare the following section between two documents:

Section: {section}

Document 1 ({doc1_name}):
{doc1_content}

Document 2 ({doc2_name}):
{doc2_content}

Analyze and provide detailed comparison results."""

            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = self.llm.invoke(messages)
                
                # Parse the comparison result
                comparison_result = self._parse_comparison_response(
                    response.content, section, doc1_content, doc2_content, doc1_name, doc2_name
                )
                comparison_results.append(comparison_result)
                
            except Exception as e:
                logger.error(f"Error comparing section {section}: {str(e)}")
                comparison_results.append({
                    'Section': section,
                    'Difference_Type': 'ERROR',
                    'Description': f'Error during comparison: {str(e)}',
                    'Document_1_Content': self.truncate_content_smart(doc1_content),
                    'Document_2_Content': self.truncate_content_smart(doc2_content),
                    'Impact_Level': 'UNKNOWN',
                    'Document_1_Name': doc1_name,
                    'Document_2_Name': doc2_name
                })
        
        return comparison_results
    
    def _parse_comparison_response(self, response_content: str, section: str, doc1_content: str, 
                                 doc2_content: str, doc1_name: str, doc2_name: str) -> Dict:
        """Parse LLM comparison response into structured format."""
        
        # Extract key information from the response
        difference_type = "MODIFIED"  # Default
        impact_level = "MEDIUM"  # Default
        
        # Simple parsing logic - can be enhanced
        if "NO_DIFFERENCE" in response_content.upper() or "IDENTICAL" in response_content.upper():
            difference_type = "NO_DIFFERENCE"
            impact_level = "LOW"
        elif "MISSING" in response_content.upper():
            if doc1_content == "NOT FOUND":
                difference_type = "MISSING_IN_DOC1"
            elif doc2_content == "NOT FOUND":
                difference_type = "MISSING_IN_DOC2"
            impact_level = "HIGH"
        elif "ADDED" in response_content.upper():
            difference_type = "ADDED"
            impact_level = "MEDIUM"
        elif "REMOVED" in response_content.upper():
            difference_type = "REMOVED"
            impact_level = "MEDIUM"
        
        # Determine impact level
        if "HIGH" in response_content.upper():
            impact_level = "HIGH"
        elif "LOW" in response_content.upper():
            impact_level = "LOW"
        
        return {
            'Section': section,
            'Difference_Type': difference_type,
            'Description': response_content.strip(),
            'Document_1_Content': self.truncate_content_smart(doc1_content, first_words=500, last_words=100),
            'Document_2_Content': self.truncate_content_smart(doc2_content, first_words=500, last_words=100),
            'Impact_Level': impact_level,
            'Document_1_Name': doc1_name,
            'Document_2_Name': doc2_name,
            'Comparison_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def create_excel_report(self, comparison_results: List[Dict], doc1_name: str, doc2_name: str) -> bytes:
        """Create Excel report from comparison results."""
        
        # Create DataFrame
        df = pd.DataFrame(comparison_results)
        
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
                    'Sections with No Differences',
                    'High Impact Changes',
                    'Medium Impact Changes',
                    'Low Impact Changes',
                    'Missing Sections'
                ],
                'Count': [
                    len(comparison_results),
                    len([r for r in comparison_results if r['Difference_Type'] != 'NO_DIFFERENCE']),
                    len([r for r in comparison_results if r['Difference_Type'] == 'NO_DIFFERENCE']),
                    len([r for r in comparison_results if r['Impact_Level'] == 'HIGH']),
                    len([r for r in comparison_results if r['Impact_Level'] == 'MEDIUM']),
                    len([r for r in comparison_results if r['Impact_Level'] == 'LOW']),
                    len([r for r in comparison_results if 'MISSING' in r['Difference_Type']])
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Document info sheet
            doc_info = pd.DataFrame({
                'Document': [doc1_name, doc2_name],
                'Comparison_Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 2,
                'Sections_Analyzed': [', '.join(self.target_sections)] * 2
            })
            doc_info.to_excel(writer, sheet_name='Document_Info', index=False)
        
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
        
        st.markdown("---")
        st.markdown("**Content Display:**")
        st.markdown("• First 500 words")
        st.markdown("• Last 100 words")
        st.markdown("• Content truncation indicator")
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Document 1")
        doc1_file = st.file_uploader(
            "Upload first PDF document",
            type=['pdf'],
            key="doc1"
        )
        
        if doc1_file:
            st.success(f"✅ {doc1_file.name} uploaded successfully")
    
    with col2:
        st.subheader("📄 Document 2")
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
                
                # Filter sections using LLM
                st.info("🤖 Filtering sections using AI...")
                doc1_sections = comparer.filter_sections_with_llm(doc1_text, doc1_file.name)
                doc2_sections = comparer.filter_sections_with_llm(doc2_text, doc2_file.name)
                
                # Compare documents using LLM
                st.info("🔍 Comparing documents using AI...")
                comparison_results = comparer.compare_documents_with_llm(
                    doc1_sections, doc2_sections, doc1_file.name, doc2_file.name
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
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sections = len(comparison_results)
                st.metric("Total Sections", total_sections)
            
            with col2:
                differences = len([r for r in comparison_results if r['Difference_Type'] != 'NO_DIFFERENCE'])
                st.metric("Sections with Differences", differences)
            
            with col3:
                high_impact = len([r for r in comparison_results if r['Impact_Level'] == 'HIGH'])
                st.metric("High Impact Changes", high_impact)
            
            with col4:
                missing_sections = len([r for r in comparison_results if 'MISSING' in r['Difference_Type']])
                st.metric("Missing Sections", missing_sections)
            
            # Detailed results table
            st.subheader("📋 Detailed Comparison")
            df_display = pd.DataFrame(comparison_results)
            st.dataframe(df_display, use_container_width=True)
            
            # Download Excel report
            st.subheader("📥 Download Report")
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"document_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
