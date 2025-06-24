import streamlit as st
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import io
import json
from datetime import datetime
import difflib
import re
from typing import List, Dict, Tuple
import base64

# Page configuration
st.set_page_config(
    page_title="Document Comparer",
    page_icon="📊",
    layout="wide"
)

class DocumentComparer:
    def __init__(self, azure_openai_api_key: str, azure_endpoint: str, deployment_name: str, api_version: str = "2023-12-01-preview"):
        """Initialize the Document Comparer with Azure OpenAI configuration."""
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_openai_api_key,
            api_version=api_version,
            deployment_name=deployment_name,
            temperature=0.1,
            max_tokens=4000
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better comparison."""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks."""
        return self.text_splitter.split_text(text)
    
    def compare_documents_ai(self, doc1_text: str, doc2_text: str, comparison_type: str = "comprehensive") -> Dict:
        """Use AI to intelligently compare two documents."""
        
        system_prompt = f"""You are an expert document analyst. Compare the two documents and identify differences in a structured way.

        Focus on the following aspects based on the comparison type '{comparison_type}':
        - Content additions, deletions, and modifications
        - Structural changes
        - Factual differences
        - Policy or procedural changes
        - Data/numerical differences
        - Semantic differences (same meaning expressed differently)

        Return your analysis in the following JSON format:
        {{
            "summary": "Brief overview of key differences",
            "total_differences": number,
            "categories": {{
                "additions": [
                    {{"section": "section name", "content": "added content", "significance": "high/medium/low"}}
                ],
                "deletions": [
                    {{"section": "section name", "content": "deleted content", "significance": "high/medium/low"}}
                ],
                "modifications": [
                    {{"section": "section name", "original": "original text", "modified": "modified text", "significance": "high/medium/low"}}
                ],
                "structural_changes": [
                    {{"type": "change type", "description": "description of change", "significance": "high/medium/low"}}
                ]
            }},
            "recommendations": ["recommendation 1", "recommendation 2"]
        }}
        """
        
        human_prompt = f"""
        Document 1:
        {doc1_text[:8000]}  # Limit text to avoid token limits
        
        Document 2:
        {doc2_text[:8000]}  # Limit text to avoid token limits
        
        Please provide a detailed comparison following the JSON format specified.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Try to parse JSON response
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response from the text
                return {
                    "summary": "AI analysis completed but response format was not structured",
                    "total_differences": 0,
                    "categories": {
                        "additions": [],
                        "deletions": [],
                        "modifications": [],
                        "structural_changes": []
                    },
                    "ai_response": response.content,
                    "recommendations": []
                }
        except Exception as e:
            st.error(f"Error in AI comparison: {str(e)}")
            return None
    
    def basic_text_comparison(self, text1: str, text2: str) -> List[Dict]:
        """Perform basic text comparison using difflib."""
        differences = []
        
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        
        diff = difflib.unified_diff(
            lines1, lines2,
            fromfile='Document 1',
            tofile='Document 2',
            lineterm=''
        )
        
        for line in diff:
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                continue
            elif line.startswith('-'):
                differences.append({
                    'type': 'deletion',
                    'content': line[1:],
                    'line_number': len(differences) + 1
                })
            elif line.startswith('+'):
                differences.append({
                    'type': 'addition',
                    'content': line[1:],
                    'line_number': len(differences) + 1
                })
        
        return differences
    
    def create_excel_report(self, ai_results: Dict, basic_results: List[Dict], doc1_name: str, doc2_name: str) -> io.BytesIO:
        """Create an Excel report with comparison results."""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            if ai_results:
                summary_data = {
                    'Metric': ['Total Differences', 'Additions', 'Deletions', 'Modifications', 'Structural Changes'],
                    'Count': [
                        ai_results.get('total_differences', 0),
                        len(ai_results.get('categories', {}).get('additions', [])),
                        len(ai_results.get('categories', {}).get('deletions', [])),
                        len(ai_results.get('categories', {}).get('modifications', [])),
                        len(ai_results.get('categories', {}).get('structural_changes', []))
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # AI Analysis Details
                if ai_results.get('categories'):
                    categories = ai_results['categories']
                    
                    # Additions
                    if categories.get('additions'):
                        additions_df = pd.DataFrame(categories['additions'])
                        additions_df.to_excel(writer, sheet_name='Additions', index=False)
                    
                    # Deletions
                    if categories.get('deletions'):
                        deletions_df = pd.DataFrame(categories['deletions'])
                        deletions_df.to_excel(writer, sheet_name='Deletions', index=False)
                    
                    # Modifications
                    if categories.get('modifications'):
                        modifications_df = pd.DataFrame(categories['modifications'])
                        modifications_df.to_excel(writer, sheet_name='Modifications', index=False)
                    
                    # Structural Changes
                    if categories.get('structural_changes'):
                        structural_df = pd.DataFrame(categories['structural_changes'])
                        structural_df.to_excel(writer, sheet_name='Structural Changes', index=False)
            
            # Basic comparison results
            if basic_results:
                basic_df = pd.DataFrame(basic_results)
                basic_df.to_excel(writer, sheet_name='Basic Comparison', index=False)
            
            # Metadata
            metadata = {
                'Property': ['Document 1', 'Document 2', 'Comparison Date', 'Analysis Type'],
                'Value': [doc1_name, doc2_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'AI + Basic Text Comparison']
            }
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        output.seek(0)
        return output

def main():
    st.title("📊 Intelligent Document Comparer")
    st.markdown("Compare two PDF documents using AI-powered analysis and export results to Excel")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Azure OpenAI Configuration
        st.subheader("Azure OpenAI Settings")
        azure_api_key = st.text_input("Azure OpenAI API Key", type="password")
        azure_endpoint = st.text_input("Azure Endpoint", placeholder="https://your-resource.openai.azure.com/")
        deployment_name = st.text_input("Deployment Name", placeholder="gpt-4")
        api_version = st.text_input("API Version", value="2023-12-01-preview")
        
        # Comparison settings
        st.subheader("Comparison Settings")
        comparison_type = st.selectbox(
            "Comparison Type",
            ["comprehensive", "content-focused", "structural", "data-focused"]
        )
        
        enable_ai_analysis = st.checkbox("Enable AI Analysis", value=True)
        enable_basic_comparison = st.checkbox("Enable Basic Text Comparison", value=True)
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Document 1")
        doc1_file = st.file_uploader("Upload first PDF document", type="pdf", key="doc1")
        
    with col2:
        st.subheader("📄 Document 2")
        doc2_file = st.file_uploader("Upload second PDF document", type="pdf", key="doc2")
    
    if st.button("🔍 Compare Documents", type="primary"):
        if not all([azure_api_key, azure_endpoint, deployment_name]):
            st.error("Please provide all Azure OpenAI configuration details in the sidebar.")
            return
        
        if not doc1_file or not doc2_file:
            st.error("Please upload both PDF documents.")
            return
        
        try:
            # Initialize the comparer
            comparer = DocumentComparer(
                azure_openai_api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                deployment_name=deployment_name,
                api_version=api_version
            )
            
            with st.spinner("Extracting text from documents..."):
                # Extract text from PDFs
                doc1_text = comparer.extract_text_from_pdf(doc1_file)
                doc2_text = comparer.extract_text_from_pdf(doc2_file)
                
                if not doc1_text or not doc2_text:
                    st.error("Failed to extract text from one or both documents.")
                    return
                
                # Preprocess text
                doc1_text = comparer.preprocess_text(doc1_text)
                doc2_text = comparer.preprocess_text(doc2_text)
            
            st.success("✅ Text extraction completed!")
            
            # Show document statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Doc 1 Length", f"{len(doc1_text):,} chars")
            with col2:
                st.metric("Doc 2 Length", f"{len(doc2_text):,} chars")
            with col3:
                st.metric("Doc 1 Words", f"{len(doc1_text.split()):,}")
            with col4:
                st.metric("Doc 2 Words", f"{len(doc2_text.split()):,}")
            
            ai_results = None
            basic_results = []
            
            # AI Analysis
            if enable_ai_analysis:
                with st.spinner("Performing AI-powered analysis..."):
                    ai_results = comparer.compare_documents_ai(doc1_text, doc2_text, comparison_type)
                
                if ai_results:
                    st.success("✅ AI analysis completed!")
                    
                    # Display AI results
                    st.subheader("🤖 AI Analysis Results")
                    
                    if ai_results.get('summary'):
                        st.info(f"**Summary:** {ai_results['summary']}")
                    
                    if ai_results.get('total_differences'):
                        st.metric("Total Differences Found", ai_results['total_differences'])
                    
                    # Display categories
                    if ai_results.get('categories'):
                        categories = ai_results['categories']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Additions", len(categories.get('additions', [])))
                        with col2:
                            st.metric("Deletions", len(categories.get('deletions', [])))
                        with col3:
                            st.metric("Modifications", len(categories.get('modifications', [])))
                        with col4:
                            st.metric("Structural Changes", len(categories.get('structural_changes', [])))
                    
                    # Show recommendations
                    if ai_results.get('recommendations'):
                        st.subheader("💡 Recommendations")
                        for i, rec in enumerate(ai_results['recommendations'], 1):
                            st.write(f"{i}. {rec}")
            
            # Basic comparison
            if enable_basic_comparison:
                with st.spinner("Performing basic text comparison..."):
                    basic_results = comparer.basic_text_comparison(doc1_text, doc2_text)
                
                st.success("✅ Basic comparison completed!")
                st.metric("Basic Differences Found", len(basic_results))
            
            # Generate Excel report
            with st.spinner("Generating Excel report..."):
                excel_buffer = comparer.create_excel_report(
                    ai_results, 
                    basic_results, 
                    doc1_file.name, 
                    doc2_file.name
                )
            
            st.success("✅ Excel report generated!")
            
            # Download button
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_buffer.getvalue(),
                file_name=f"document_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Display preview of differences
            if ai_results and ai_results.get('categories'):
                st.subheader("📋 Preview of Key Differences")
                
                # Show top modifications
                modifications = ai_results['categories'].get('modifications', [])
                if modifications:
                    st.write("**Top Modifications:**")
                    for i, mod in enumerate(modifications[:3], 1):
                        with st.expander(f"Modification {i} - {mod.get('section', 'Unknown Section')}"):
                            st.write(f"**Original:** {mod.get('original', 'N/A')}")
                            st.write(f"**Modified:** {mod.get('modified', 'N/A')}")
                            st.write(f"**Significance:** {mod.get('significance', 'N/A')}")
                
                # Show top additions
                additions = ai_results['categories'].get('additions', [])
                if additions:
                    st.write("**Top Additions:**")
                    for i, add in enumerate(additions[:3], 1):
                        with st.expander(f"Addition {i} - {add.get('section', 'Unknown Section')}"):
                            st.write(f"**Content:** {add.get('content', 'N/A')}")
                            st.write(f"**Significance:** {add.get('significance', 'N/A')}")
        
        except Exception as e:
            st.error(f"An error occurred during comparison: {str(e)}")
            st.error("Please check your Azure OpenAI configuration and try again.")

if __name__ == "__main__":
    # Install required packages info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Required packages:**")
    st.sidebar.code("""
    pip install streamlit
    pip install langchain-openai
    pip install PyPDF2
    pip install pandas
    pip install openpyxl
    """)
    
    main()
