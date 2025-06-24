import streamlit as st
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import io
import json
from datetime import datetime
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
            chunk_size=3000,
            chunk_overlap=300,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num} ---\n{page_text}\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def parse_document_structure(self, text: str, doc_name: str) -> Dict:
        """Use LLM to parse and understand document structure."""
        
        system_prompt = """You are an expert document analyst. Analyze the given document and extract its structure, key information, and content organization.

        Return your analysis in the following JSON format:
        {
            "document_type": "type of document (contract, report, policy, etc.)",
            "title": "document title if available",
            "sections": [
                {
                    "section_name": "name of section",
                    "page_numbers": [1, 2],
                    "key_points": ["point 1", "point 2"],
                    "content_summary": "brief summary of section content",
                    "data_elements": ["any numbers, dates, or structured data"],
                    "importance_level": "high/medium/low"
                }
            ],
            "key_entities": {
                "people": ["person names"],
                "organizations": ["organization names"],
                "dates": ["important dates"],
                "numbers": ["significant numbers or amounts"],
                "locations": ["places mentioned"]
            },
            "document_metadata": {
                "total_pages": number,
                "estimated_word_count": number,
                "language": "language of document",
                "formality_level": "formal/informal/technical"
            },
            "main_topics": ["topic 1", "topic 2", "topic 3"],
            "document_purpose": "brief description of document purpose"
        }
        """
        
        human_prompt = f"""
        Please analyze the following document titled "{doc_name}":

        {text[:12000]}  # Increased limit for better analysis
        
        Provide a comprehensive structural analysis following the JSON format specified.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # Fallback structure if JSON parsing fails
                return {
                    "document_type": "unknown",
                    "title": doc_name,
                    "sections": [],
                    "key_entities": {},
                    "document_metadata": {},
                    "main_topics": [],
                    "document_purpose": "Could not parse document structure",
                    "raw_analysis": response.content
                }
        except Exception as e:
            st.error(f"Error in document parsing: {str(e)}")
            return None
    
    def intelligent_document_comparison(self, doc1_structure: Dict, doc2_structure: Dict, doc1_text: str, doc2_text: str) -> Dict:
        """Use LLM to perform intelligent comparison between two parsed documents."""
        
        system_prompt = """You are an expert document comparison analyst. Compare two documents that have been pre-analyzed for structure and content. 

        Perform a comprehensive comparison focusing on:
        1. Structural differences (sections added/removed/reorganized)
        2. Content changes (additions, deletions, modifications)
        3. Entity changes (people, dates, numbers, organizations)
        4. Policy/procedural changes
        5. Semantic differences (same meaning expressed differently)
        6. Data/numerical changes with impact analysis
        7. Compliance or regulatory implications
        8. Risk assessment of changes

        Return your analysis in the following JSON format:
        {
            "comparison_summary": {
                "total_differences": number,
                "severity_assessment": "high/medium/low",
                "change_type": "major_revision/minor_updates/cosmetic_changes",
                "overall_impact": "description of overall impact"
            },
            "structural_differences": [
                {
                    "change_type": "section_added/section_removed/section_modified/section_reordered",
                    "section_name": "name of affected section",
                    "description": "detailed description of change",
                    "impact_level": "high/medium/low",
                    "page_reference": "page numbers affected"
                }
            ],
            "content_differences": [
                {
                    "category": "addition/deletion/modification/replacement",
                    "section": "section where change occurred",
                    "original_content": "original text (for modifications)",
                    "new_content": "new text (for additions/modifications)",
                    "context": "surrounding context",
                    "significance": "high/medium/low",
                    "impact_description": "explanation of why this change matters",
                    "page_reference": "page numbers"
                }
            ],
            "entity_changes": [
                {
                    "entity_type": "person/organization/date/number/location",
                    "change_type": "added/removed/modified",
                    "original_value": "original value if modified",
                    "new_value": "new value if added/modified",
                    "context": "context where entity appears",
                    "impact": "impact of this entity change"
                }
            ],
            "policy_changes": [
                {
                    "policy_area": "area of policy change",
                    "change_description": "description of policy change",
                    "old_policy": "previous policy if applicable",
                    "new_policy": "new policy",
                    "compliance_impact": "impact on compliance",
                    "stakeholder_impact": "who is affected by this change"
                }
            ],
            "data_changes": [
                {
                    "data_type": "financial/statistical/technical/other",
                    "field_name": "name of data field",
                    "original_value": "original value",
                    "new_value": "new value",
                    "percentage_change": "percentage change if applicable",
                    "context": "context of the data",
                    "business_impact": "impact on business/operations"
                }
            ],
            "recommendations": [
                {
                    "priority": "high/medium/low",
                    "action": "recommended action",
                    "rationale": "why this action is recommended",
                    "stakeholders": ["affected stakeholders"]
                }
            ],
            "risk_assessment": {
                "legal_risks": ["potential legal risks"],
                "operational_risks": ["potential operational risks"],
                "compliance_risks": ["potential compliance risks"],
                "financial_risks": ["potential financial risks"]
            }
        }
        """
        
        human_prompt = f"""
        Compare these two documents:

        DOCUMENT 1 STRUCTURE:
        {json.dumps(doc1_structure, indent=2)}

        DOCUMENT 1 CONTENT (First 8000 characters):
        {doc1_text[:8000]}

        DOCUMENT 2 STRUCTURE:
        {json.dumps(doc2_structure, indent=2)}

        DOCUMENT 2 CONTENT (First 8000 characters):
        {doc2_text[:8000]}

        Provide a comprehensive comparison analysis following the JSON format specified. Focus on meaningful differences that would impact understanding, compliance, or operations.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                return {
                    "comparison_summary": {
                        "total_differences": 0,
                        "severity_assessment": "unknown",
                        "change_type": "analysis_failed",
                        "overall_impact": "Could not parse comparison results"
                    },
                    "raw_analysis": response.content,
                    "structural_differences": [],
                    "content_differences": [],
                    "entity_changes": [],
                    "policy_changes": [],
                    "data_changes": [],
                    "recommendations": [],
                    "risk_assessment": {}
                }
        except Exception as e:
            st.error(f"Error in intelligent comparison: {str(e)}")
            return None
    
    def chunk_and_compare_sections(self, doc1_text: str, doc2_text: str) -> List[Dict]:
        """Compare documents in chunks for detailed section-by-section analysis."""
        
        doc1_chunks = self.text_splitter.split_text(doc1_text)
        doc2_chunks = self.text_splitter.split_text(doc2_text)
        
        chunk_comparisons = []
        
        max_chunks = max(len(doc1_chunks), len(doc2_chunks))
        
        for i in range(max_chunks):
            chunk1 = doc1_chunks[i] if i < len(doc1_chunks) else ""
            chunk2 = doc2_chunks[i] if i < len(doc2_chunks) else ""
            
            if chunk1 or chunk2:  # Only compare if at least one chunk has content
                comparison = self.compare_text_chunks(chunk1, chunk2, i + 1)
                if comparison:
                    chunk_comparisons.append(comparison)
        
        return chunk_comparisons
    
    def compare_text_chunks(self, chunk1: str, chunk2: str, chunk_number: int) -> Dict:
        """Compare two text chunks using LLM."""
        
        system_prompt = """You are analyzing two text chunks from similar documents. Identify any meaningful differences between them.

        Return your analysis in JSON format:
        {
            "has_differences": true/false,
            "differences": [
                {
                    "type": "addition/deletion/modification/semantic_change",
                    "description": "description of the difference",
                    "impact": "high/medium/low",
                    "chunk1_content": "relevant content from chunk 1",
                    "chunk2_content": "relevant content from chunk 2"
                }
            ],
            "similarity_score": "percentage similarity (0-100)"
        }
        """
        
        human_prompt = f"""
        Compare these two text chunks:

        CHUNK 1:
        {chunk1}

        CHUNK 2:
        {chunk2}

        Identify meaningful differences and provide similarity assessment.
        """
        
        try:
            if not chunk1 and not chunk2:
                return None
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                result = json.loads(response.content)
                result["chunk_number"] = chunk_number
                return result
            except json.JSONDecodeError:
                return {
                    "chunk_number": chunk_number,
                    "has_differences": True,
                    "differences": [],
                    "similarity_score": "unknown",
                    "raw_analysis": response.content
                }
        except Exception as e:
            return None
    
    def create_comprehensive_excel_report(self, doc1_structure: Dict, doc2_structure: Dict, 
                                        comparison_results: Dict, chunk_comparisons: List[Dict],
                                        doc1_name: str, doc2_name: str) -> io.BytesIO:
        """Create a comprehensive Excel report with all analysis results."""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Executive Summary
            if comparison_results:
                summary_data = []
                comp_summary = comparison_results.get('comparison_summary', {})
                
                summary_data.append(['Total Differences', comp_summary.get('total_differences', 'N/A')])
                summary_data.append(['Severity Assessment', comp_summary.get('severity_assessment', 'N/A')])
                summary_data.append(['Change Type', comp_summary.get('change_type', 'N/A')])
                summary_data.append(['Overall Impact', comp_summary.get('overall_impact', 'N/A')])
                
                # Count different types of changes
                summary_data.append(['Structural Differences', len(comparison_results.get('structural_differences', []))])
                summary_data.append(['Content Differences', len(comparison_results.get('content_differences', []))])
                summary_data.append(['Entity Changes', len(comparison_results.get('entity_changes', []))])
                summary_data.append(['Policy Changes', len(comparison_results.get('policy_changes', []))])
                summary_data.append(['Data Changes', len(comparison_results.get('data_changes', []))])
                
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Document Structures Comparison
            if doc1_structure and doc2_structure:
                structure_comparison = []
                
                # Document metadata comparison
                doc1_meta = doc1_structure.get('document_metadata', {})
                doc2_meta = doc2_structure.get('document_metadata', {})
                
                structure_comparison.append(['Document Type', 
                                           doc1_structure.get('document_type', 'N/A'),
                                           doc2_structure.get('document_type', 'N/A')])
                structure_comparison.append(['Total Pages', 
                                           doc1_meta.get('total_pages', 'N/A'),
                                           doc2_meta.get('total_pages', 'N/A')])
                structure_comparison.append(['Estimated Word Count', 
                                           doc1_meta.get('estimated_word_count', 'N/A'),
                                           doc2_meta.get('estimated_word_count', 'N/A')])
                structure_comparison.append(['Main Topics', 
                                           ', '.join(doc1_structure.get('main_topics', [])),
                                           ', '.join(doc2_structure.get('main_topics', []))])
                
                structure_df = pd.DataFrame(structure_comparison, 
                                          columns=['Attribute', 'Document 1', 'Document 2'])
                structure_df.to_excel(writer, sheet_name='Document Structure', index=False)
            
            # Content Differences
            if comparison_results and comparison_results.get('content_differences'):
                content_df = pd.DataFrame(comparison_results['content_differences'])
                content_df.to_excel(writer, sheet_name='Content Differences', index=False)
            
            # Structural Differences
            if comparison_results and comparison_results.get('structural_differences'):
                structural_df = pd.DataFrame(comparison_results['structural_differences'])
                structural_df.to_excel(writer, sheet_name='Structural Changes', index=False)
            
            # Entity Changes
            if comparison_results and comparison_results.get('entity_changes'):
                entity_df = pd.DataFrame(comparison_results['entity_changes'])
                entity_df.to_excel(writer, sheet_name='Entity Changes', index=False)
            
            # Policy Changes
            if comparison_results and comparison_results.get('policy_changes'):
                policy_df = pd.DataFrame(comparison_results['policy_changes'])
                policy_df.to_excel(writer, sheet_name='Policy Changes', index=False)
            
            # Data Changes
            if comparison_results and comparison_results.get('data_changes'):
                data_df = pd.DataFrame(comparison_results['data_changes'])
                data_df.to_excel(writer, sheet_name='Data Changes', index=False)
            
            # Recommendations
            if comparison_results and comparison_results.get('recommendations'):
                recommendations_df = pd.DataFrame(comparison_results['recommendations'])
                recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Risk Assessment
            if comparison_results and comparison_results.get('risk_assessment'):
                risk_data = []
                risk_assessment = comparison_results['risk_assessment']
                
                for risk_type, risks in risk_assessment.items():
                    for risk in risks:
                        risk_data.append([risk_type.replace('_', ' ').title(), risk])
                
                if risk_data:
                    risk_df = pd.DataFrame(risk_data, columns=['Risk Type', 'Risk Description'])
                    risk_df.to_excel(writer, sheet_name='Risk Assessment', index=False)
            
            # Chunk-by-Chunk Analysis
            if chunk_comparisons:
                chunk_data = []
                for chunk_comp in chunk_comparisons:
                    if chunk_comp and chunk_comp.get('has_differences'):
                        for diff in chunk_comp.get('differences', []):
                            chunk_data.append([
                                chunk_comp.get('chunk_number'),
                                diff.get('type'),
                                diff.get('description'),
                                diff.get('impact'),
                                chunk_comp.get('similarity_score')
                            ])
                
                if chunk_data:
                    chunk_df = pd.DataFrame(chunk_data, 
                                          columns=['Chunk Number', 'Change Type', 'Description', 
                                                 'Impact', 'Similarity Score'])
                    chunk_df.to_excel(writer, sheet_name='Detailed Analysis', index=False)
            
            # Metadata
            metadata = [
                ['Document 1', doc1_name],
                ['Document 2', doc2_name],
                ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Analysis Type', 'AI-Powered Intelligent Comparison'],
                ['Comparison Method', 'LLM-based structural and content analysis']
            ]
            metadata_df = pd.DataFrame(metadata, columns=['Property', 'Value'])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        output.seek(0)
        return output

def main():
    st.title("🤖 AI-Powered Document Comparer")
    st.markdown("Intelligent document comparison using advanced LLM analysis - No basic text diff, only AI insights!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Azure OpenAI Configuration
        st.subheader("Azure OpenAI Settings")
        azure_api_key = st.text_input("Azure OpenAI API Key", type="password")
        azure_endpoint = st.text_input("Azure Endpoint", placeholder="https://your-resource.openai.azure.com/")
        deployment_name = st.text_input("Deployment Name", placeholder="gpt-4")
        api_version = st.text_input("API Version", value="2023-12-01-preview")
        
        # Analysis settings
        st.subheader("Analysis Settings")
        enable_chunk_analysis = st.checkbox("Enable Detailed Chunk Analysis", value=True, 
                                           help="Performs section-by-section detailed comparison")
        
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["comprehensive", "focused", "executive_summary"],
            help="Choose the depth of analysis"
        )
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Document 1")
        doc1_file = st.file_uploader("Upload first PDF document", type="pdf", key="doc1")
        
    with col2:
        st.subheader("📄 Document 2")
        doc2_file = st.file_uploader("Upload second PDF document", type="pdf", key="doc2")
    
    if st.button("🧠 Analyze Documents with AI", type="primary"):
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
            
            # Step 1: Extract text from PDFs
            with st.spinner("📝 Extracting text from documents..."):
                doc1_text = comparer.extract_text_from_pdf(doc1_file)
                doc2_text = comparer.extract_text_from_pdf(doc2_file)
                
                if not doc1_text or not doc2_text:
                    st.error("Failed to extract text from one or both documents.")
                    return
            
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
            
            # Step 2: Parse document structures using AI
            with st.spinner("🔍 AI is analyzing document structures..."):
                doc1_structure = comparer.parse_document_structure(doc1_text, doc1_file.name)
                doc2_structure = comparer.parse_document_structure(doc2_text, doc2_file.name)
            
            if not doc1_structure or not doc2_structure:
                st.error("Failed to parse document structures.")
                return
            
            st.success("✅ Document structure analysis completed!")
            
            # Display document insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Document 1 Insights")
                if doc1_structure.get('document_type'):
                    st.info(f"**Type:** {doc1_structure['document_type']}")
                if doc1_structure.get('main_topics'):
                    st.write(f"**Main Topics:** {', '.join(doc1_structure['main_topics'][:3])}")
                if doc1_structure.get('sections'):
                    st.write(f"**Sections Found:** {len(doc1_structure['sections'])}")
            
            with col2:
                st.subheader("📊 Document 2 Insights")
                if doc2_structure.get('document_type'):
                    st.info(f"**Type:** {doc2_structure['document_type']}")
                if doc2_structure.get('main_topics'):
                    st.write(f"**Main Topics:** {', '.join(doc2_structure['main_topics'][:3])}")
                if doc2_structure.get('sections'):
                    st.write(f"**Sections Found:** {len(doc2_structure['sections'])}")
            
            # Step 3: Intelligent comparison
            with st.spinner("🧠 Performing intelligent AI comparison..."):
                comparison_results = comparer.intelligent_document_comparison(
                    doc1_structure, doc2_structure, doc1_text, doc2_text
                )
            
            if not comparison_results:
                st.error("Failed to perform intelligent comparison.")
                return
            
            st.success("✅ AI comparison completed!")
            
            # Display comparison results
            if comparison_results.get('comparison_summary'):
                summary = comparison_results['comparison_summary']
                
                st.subheader("📋 Comparison Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Differences", summary.get('total_differences', 0))
                with col2:
                    severity = summary.get('severity_assessment', 'unknown')
                    st.metric("Severity", severity.upper())
                with col3:
                    st.metric("Change Type", summary.get('change_type', 'unknown').replace('_', ' ').title())
                with col4:
                    content_changes = len(comparison_results.get('content_differences', []))
                    st.metric("Content Changes", content_changes)
                
                if summary.get('overall_impact'):
                    st.info(f"**Overall Impact:** {summary['overall_impact']}")
            
            # Step 4: Detailed chunk analysis (if enabled)
            chunk_comparisons = []
            if enable_chunk_analysis:
                with st.spinner("🔍 Performing detailed section-by-section analysis..."):
                    chunk_comparisons = comparer.chunk_and_compare_sections(doc1_text, doc2_text)
                
                if chunk_comparisons:
                    differences_count = sum(1 for chunk in chunk_comparisons if chunk and chunk.get('has_differences'))
                    st.success(f"✅ Detailed analysis completed! Found differences in {differences_count} sections.")
            
            # Display key findings
            st.subheader("🔍 Key Findings")
            
            # Content differences
            if comparison_results.get('content_differences'):
                st.write("**📝 Content Changes:**")
                for i, diff in enumerate(comparison_results['content_differences'][:3], 1):
                    with st.expander(f"Change {i}: {diff.get('category', 'Unknown').title()} - {diff.get('significance', 'Unknown')} Impact"):
                        st.write(f"**Section:** {diff.get('section', 'Unknown')}")
                        if diff.get('original_content'):
                            st.write(f"**Original:** {diff.get('original_content')}")
                        if diff.get('new_content'):
                            st.write(f"**New:** {diff.get('new_content')}")
                        if diff.get('impact_description'):
                            st.write(f"**Impact:** {diff.get('impact_description')}")
            
            # Policy changes
            if comparison_results.get('policy_changes'):
                st.write("**📋 Policy Changes:**")
                for i, policy in enumerate(comparison_results['policy_changes'][:2], 1):
                    with st.expander(f"Policy Change {i}: {policy.get('policy_area', 'Unknown Area')}"):
                        st.write(f"**Description:** {policy.get('change_description')}")
                        if policy.get('compliance_impact'):
                            st.write(f"**Compliance Impact:** {policy.get('compliance_impact')}")
            
            # Recommendations
            if comparison_results.get('recommendations'):
                st.write("**💡 AI Recommendations:**")
                for i, rec in enumerate(comparison_results['recommendations'][:3], 1):
                    priority = rec.get('priority', 'medium')
                    color = "🔴" if priority == 'high' else "🟡" if priority == 'medium' else "🟢"
                    st.write(f"{color} **{rec.get('action')}** - {rec.get('rationale')}")
            
            # Step 5: Generate comprehensive Excel report
            with st.spinner("📊 Generating comprehensive Excel report..."):
                excel_buffer = comparer.create_comprehensive_excel_report(
                    doc1_structure, doc2_structure, comparison_results, chunk_comparisons,
                    doc1_file.name, doc2_file.name
                )
            
            st.success("✅ Comprehensive Excel report generated!")
            
            # Download button
            st.download_button(
                label="📥 Download Comprehensive Analysis Report",
                data=excel_buffer.getvalue(),
                file_name=f"ai_document_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Risk assessment display
            if comparison_results.get('risk_assessment'):
                st.subheader("⚠️ Risk Assessment")
                risk_assessment = comparison_results['risk_assessment']
                
                risk_cols = st.columns(2)
                col_idx = 0
                
                for risk_type, risks in risk_assessment.items():
                    if risks:
                        with risk_cols[col_idx % 2]:
                            st.write(f"**{risk_type.replace('_', ' ').title()}:**")
                            for risk in risks[:2]:  # Show first 2 risks of each type
                                st.write(f"• {risk}")
                        col_idx += 1
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
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
