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

    def estimate_token_count(self, text: str) -> int:
        return len(text) // 4

    def chunk_document(self, text: str, chunk_size: int = None) -> List[str]:
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = current_pos + chunk_size
            
            if end_pos >= len(text):
                chunks.append(text[current_pos:])
                break
            
            para_break = text.rfind('\n\n', current_pos, end_pos)
            if para_break > current_pos:
                chunks.append(text[current_pos:para_break])
                current_pos = para_break + 2
                continue
            
            sent_break = max(
                text.rfind('. ', current_pos, end_pos),
                text.rfind('.\n', current_pos, end_pos),
                text.rfind('!\n', current_pos, end_pos),
                text.rfind('?\n', current_pos, end_pos)
            )
            
            if sent_break > current_pos:
                chunks.append(text[current_pos:sent_break + 1])
                current_pos = sent_break + 1
            else:
                chunks.append(text[current_pos:end_pos])
                current_pos = end_pos
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def pre_filter_sections_with_keywords(self, document_text: str) -> Dict[str, List[str]]:
        
        section_keywords = {
            "FORWARDING LETTER": [
                "sub:", "subject:", "issuance", "forwarding", "ref:", "reference",
                "disclaimer", "dispute", "english version", "dear", "regards"
            ],
            "PREAMBLE": [
                "preamble", "whereas", "now therefore", "witnesseth", "parties agree"
            ],
            "SCHEDULE": [
                "schedule", "annexure", "attachment", "exhibit", "details", "particulars"
            ],
            "DEFINITIONS & ABBREVIATIONS": [
                "definition", "abbreviation", "means", "shall mean", "interpreted", 
                "glossary", "terms", "terminology"
            ]
        }
        
        potential_sections = {}
        text_lower = document_text.lower()
        
        for section, keywords in section_keywords.items():
            potential_chunks = []
            
            for keyword in keywords:
                start_pos = 0
                while True:
                    pos = text_lower.find(keyword, start_pos)
                    if pos == -1:
                        break
                    
                    context_start = max(0, pos - 500)
                    context_end = min(len(document_text), pos + 1500)
                    context = document_text[context_start:context_end]
                    
                    potential_chunks.append(context)
                    start_pos = pos + 1
            
            potential_sections[section] = potential_chunks
        
        return potential_sections

    def filter_sections_with_llm_chunked(self, document_text: str, doc_name: str) -> Dict[str, str]:
        
        if self.estimate_token_count(document_text) < 2500:  # Conservative limit
            return self.filter_sections_with_llm_direct(document_text, doc_name)
        
        st.info(f"Document {doc_name} is large, using chunked processing...")
        
        try:
            potential_sections = self.pre_filter_sections_with_keywords(document_text)
            final_sections = {}
            
            for section in self.target_sections:
                chunks = potential_sections.get(section, [])
                
                if not chunks:
                    doc_chunks = self.chunk_document(document_text, 6000)
                    chunks = doc_chunks[:3]
                
                if chunks:
                    section_content = self.extract_section_from_chunks(chunks, section, doc_name)
                    final_sections[section] = section_content
                else:
                    final_sections[section] = "NOT FOUND"
            
            return final_sections
            
        except Exception as e:
            logger.error(f"Chunked processing failed for {doc_name}: {str(e)}")
            return self.filter_sections_sequential_chunks(document_text, doc_name)

    def filter_sections_with_llm_direct(self, document_text: str, doc_name: str) -> Dict[str, str]:
        
        system_prompt = """You are an expert document analyzer. Your task is to extract specific sections from legal/business documents and return the results as valid JSON.

Target sections and instructions:

1. FORWARDING LETTER:
   - Look for the beginning of the document where a subject line like "Sub: Issuance..." is mentioned — this usually starts the forwarding.
   - Look for end phrases like "Disclaimer: In case of dispute, English version..." or a sentence signaling end of letter.
   - Extract everything in between, even if the wording differs slightly.
   - Generally the first page

2. PREAMBLE:
   - Look for a section labeled "PREAMBLE" or similar — match approximate variants.

3. SCHEDULE:
   - Look for a section labeled "SCHEDULE" or similar — match approximate variants.

4. DEFINITIONS & ABBREVIATIONS:
   - Locate section titled "DEFINITIONS", "ABBREVIATIONS", or similar — approximate matches are fine.

Instructions:
- Do not be strict with exact matching.
- Case, small wording variations, or punctuation should not block detection.
- Return full section text.
- If nothing close is found, return "NOT FOUND".
- IMPORTANT: Return ONLY valid JSON, no explanations or additional text.
"""

        user_prompt = f"""Analyze the following document and extract the target sections. Return ONLY valid JSON with no additional text or explanations.

Document Name: {doc_name}

Document Content:
{document_text}

Return as JSON format:
{{
  "FORWARDING LETTER": "extracted content or NOT FOUND",
  "PREAMBLE": "extracted content or NOT FOUND", 
  "SCHEDULE": "extracted content or NOT FOUND",
  "DEFINITIONS & ABBREVIATIONS": "extracted content or NOT FOUND"
}}
"""
        
        try:
            llm_with_json = self.llm.bind(response_format={"type": "json_object"})
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            st.write(f"Processing {doc_name} directly (small document)...")
            
            response = llm_with_json.invoke(messages)
            
            try:
                sections = json.loads(response.content)
                st.write(f"Successfully parsed sections for {doc_name}: {list(sections.keys())}")
                return sections
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing failed for {doc_name}: {str(json_error)}")
                return self._fallback_section_extraction(document_text)
                
        except Exception as e:
            logger.error(f"Error in direct LLM processing for {doc_name}: {str(e)}")
            return self._fallback_section_extraction(document_text)

    def extract_section_from_chunks(self, chunks: List[str], section_name: str, doc_name: str) -> str:
        
        system_prompt = f"""You are an expert at finding specific sections in document chunks. 

Your task is to find the "{section_name}" section from the provided document chunks.

Section guidelines:
- FORWARDING LETTER: Look for subject lines, reference numbers, formal letter content, disclaimers
- PREAMBLE: Look for "PREAMBLE", "WHEREAS" clauses, introductory statements
- SCHEDULE: Look for "SCHEDULE", "ANNEXURE", detailed lists, tables
- DEFINITIONS & ABBREVIATIONS: Look for "DEFINITIONS", "ABBREVIATIONS", term explanations

Instructions:
1. Examine all chunks for the target section
2. If found, return the complete section content
3. If not found in any chunk, return "NOT FOUND"
4. Return ONLY the section content or "NOT FOUND" - no explanations
"""
        best_match = "NOT FOUND"
        
        for i, chunk in enumerate(chunks):
            if self.estimate_token_count(chunk) > 2500:
                # Further split if chunk is still too large
                sub_chunks = self.chunk_document(chunk, 4000)
                chunk = sub_chunks[0]  # Take first sub-chunk
            
            user_prompt = f"""Find the "{section_name}" section in this document chunk:

Document: {doc_name}
Chunk {i+1}/{len(chunks)}:

{chunk}

Return ONLY the section content or "NOT FOUND":"""
            
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = self.llm.invoke(messages)
                result = response.content.strip()
                
                if result != "NOT FOUND" and len(result) > 10:
                    best_match = result
                    break  # Found it, no need to check other chunks
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i+1} for section {section_name}: {str(e)}")
                continue
        
        return best_match

    def filter_sections_sequential_chunks(self, document_text: str, doc_name: str) -> Dict[str, str]:
        
        chunks = self.chunk_document(document_text, 6000)
        st.info(f"Processing {doc_name} in {len(chunks)} sequential chunks...")
        
        final_sections = {}
        
        for section in self.target_sections:
            section_content = "NOT FOUND"
            
            # Try to find section in chunks
            for i, chunk in enumerate(chunks):
                try:
                    if self.estimate_token_count(chunk) > 2500:
                        continue  # Skip chunks that are still too large
                    
                    result = self.extract_section_from_chunks([chunk], section, f"{doc_name}_chunk_{i+1}")
                    
                    if result != "NOT FOUND":
                        section_content = result
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1} for section {section}: {str(e)}")
                    continue
            
            final_sections[section] = section_content
        
        return final_sections

    def filter_sections_with_llm(self, document_text: str, doc_name: str) -> Dict[str, str]:
        return self.filter_sections_with_llm_chunked(document_text, doc_name)
    
    def _fallback_section_extraction(self, text: str) -> Dict[str, str]:
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

                max_widths = {
                    'A': 15,  # Samples affected
                    'B': 35,  # Observation Category
                    'C': 20,  # Page
                    'D': 50,  # Sub-category
                    'E': 80   # Content (limited width)
                }
            
                for col_letter, max_width in max_widths.items():
                    worksheet.column_dimensions[col_letter].width = max_width
            
                # Apply text wrapping to all cells
                for row in worksheet.iter_rows():
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
            
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
                
                    adjusted_width = max_length + 2
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                for row in worksheet.iter_rows():
                    for cell in row:
                        if cell.coordinate.startswith('B') and cell.row > 1:
                            # Merged cells get center alignment with wrap
                            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                        else:
                            # All other cells get wrap with top alignment
                            cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            sections_with_differences = len([r for r in comparison_results if 'Mismatch' in r.get('Observation - Category', '')])
            sections_without_differences = len(comparison_results) - sections_with_differences
        
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
                st.write(f"Filed Copy extracted text:\n{doc1_text[:2000]}")
                doc2_text = comparer.extract_text_from_pdf(doc2_file)
                
                if not doc1_text or not doc2_text:
                    st.error("❌ Failed to extract text from one or both documents")
                    return
                
                # Extract sample number from document 2 (Customer Copy)
                sample_number = comparer.extract_sample_number_from_filename(doc2_file.name)
                
                # Filter sections using LLM
                st.info("🤖 Filtering sections using AI...")
                doc1_sections = comparer.filter_sections_with_llm(doc1_text, doc1_file.name)
                logger.info("📄 Filed Copy Extracted Sections:\n" + json.dumps(doc1_sections, indent=2))
                doc2_sections = comparer.filter_sections_with_llm(doc2_text, doc2_file.name)
                logger.info("📄 Customer Copy Extracted Sections:\n" + json.dumps(doc2_sections, indent=2))
                
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
