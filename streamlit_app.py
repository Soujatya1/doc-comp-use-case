import streamlit as st
import fitz
import io
import pandas as pd
import os
from openai import AzureOpenAI
import re
import xlsxwriter

st.set_page_config(layout="wide")

EXCLUDE_SECTIONS = [
    "First Premium Receipt",
    "Your Renewal Page",
]

SECTION_HEADERS = [
    "FORWARDING LETTER", "PREAMBLE", "SCHEDULE", "DEFINITIONS & ABBREVIATIONS",
    "PART C", "PART D", "PART F", "PART G", "ANNEXURE AA", "ANNEXURE BB", "ANNEXURE CC"
]

st.sidebar.header("🔧 Azure OpenAI Configuration")

azure_endpoint = st.sidebar.text_input(
    "Azure OpenAI Endpoint", 
    value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    help="Your Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/)",
    placeholder="https://your-resource.openai.azure.com/"
)

azure_api_key = st.sidebar.text_input(
    "Azure OpenAI API Key", 
    value=os.getenv("AZURE_OPENAI_API_KEY", ""),
    type="password",
    help="Your Azure OpenAI API key"
)

azure_api_version = st.sidebar.text_input(
    "API Version", 
    value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    help="Azure OpenAI API version"
)

deployment_name = st.sidebar.text_input(
    "Deployment Name", 
    value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
    help="Your Azure OpenAI deployment/model name"
)

if st.sidebar.button("Test Connection"):
    if azure_endpoint and azure_api_key:
        try:
            test_client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version
            )
            test_response = test_client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            st.sidebar.success("Connection successful!")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {str(e)}")
    else:
        st.sidebar.error("Please enter endpoint and API key")

def get_azure_client():
    if not azure_endpoint or not azure_api_key:
        return None
    
    try:
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version
        )
        return client
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI client: {str(e)}")
        return None

def is_image_only_page(page):
    text = page.get_text("text").strip()
    return not text and (page.get_images(full=True) or page.get_drawings())

def extract_final_filtered_pdf(uploaded_file_bytes, is_customer_copy=False):
    doc = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
    filtered_doc = fitz.open()
    text_summary = ""
    current_section = "UNKNOWN"

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        text_lower = text.lower()
        if is_image_only_page(page) or any(section.lower() in text_lower for section in EXCLUDE_SECTIONS):
            continue
        for header in SECTION_HEADERS:
            if header.lower() in text_lower:
                current_section = header
                break
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

def extract_product_name_and_uin(text):
    lines = text.splitlines()
    product_name = ""
    product_uin = ""
    for i, line in enumerate(lines[:50]):
        if not product_name and 5 < len(line.strip()) < 80 and not line.isupper() and any(word in line.lower() for word in ["life", "policy", "plan", "bima"]):
            product_name = line.strip()
        if not product_uin:
            match = re.search(r'UIN[:\s]*([A-Z0-9\-]{10,})', line)
            if match:
                product_uin = match.group(1).strip()
        if product_name and product_uin:
            break
    return product_name, product_uin

def extract_sample_number_from_filename(filename: str) -> str:
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

def is_header_line(line):
    return line.strip().upper() in SECTION_HEADERS

def extract_policy_name_and_type(lines, heading_index):
    search_window = lines[max(0, heading_index - 10):heading_index]
    policy_name = ""
    plan_description = ""
    for i in range(len(search_window) - 1):
        current_line = search_window[i].strip()
        next_line = search_window[i + 1].strip()
        if (
            current_line
            and len(current_line.split()) > 3
            and not current_line.isupper()
            and any(word in next_line.lower() for word in ["linked", "participating", "annuity", "plan", "deferred"])
        ):
            policy_name = current_line
            plan_description = next_line
            break
    return policy_name, plan_description

def extract_forwarding_letter_section_filed(text):
    lines = text.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "FORWARDING LETTER")
        end_idx = next(i for i, line in enumerate(lines[start_idx+1:], start=start_idx+1)
                       if is_header_line(line) and line.strip().upper() != "FORWARDING LETTER")
        policy_name, plan_type = extract_policy_name_and_type(lines, start_idx)
        section = "\n".join(lines[start_idx:end_idx]).strip()
        if policy_name and plan_type:
            section = f"Policy Name: {policy_name}\nPlan Type: {plan_type}\n\n" + section
        return section
    except StopIteration:
        return "FORWARDING LETTER section not found in Filed Copy."

def extract_forwarding_letter_section_customer(text):
    lines = text.splitlines()
    try:
        end_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "PREAMBLE")
        policy_name, plan_type = extract_policy_name_and_type(lines, end_idx)
        section = "\n".join(lines[:end_idx]).strip()
        if policy_name and plan_type:
            section = f"Policy Name: {policy_name}\nPlan Type: {plan_type}\n\n" + section
        return section
    except StopIteration:
        return "FORWARDING LETTER section not found in Customer Copy."

def extract_schedule_section(text):
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
        if "GRIEVANCE REDRESSAL MECHANISM" in line_upper or line_upper in SECTION_HEADERS and line_upper != "SCHEDULE":
            end_idx = j
            break
    end_idx = end_idx or len(lines)
    return "\n".join(lines[schedule_start:end_idx]).strip()

def extract_part_c_section(text):
    lines = text.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "DEFINITIONS & ABBREVIATIONS")
        part_c_start = None
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip().upper() == "PART C":
                part_c_start = j + 1
                break
        if part_c_start is None:
            return "PART C header not found after DEFINITIONS & ABBREVIATIONS."
    except StopIteration:
        return "DEFINITIONS & ABBREVIATIONS not found. Cannot extract PART C."
    end_idx = None
    for j in range(part_c_start, len(lines)):
        if lines[j].strip().upper() == "PART D":
            end_idx = j
            break
    end_idx = end_idx or len(lines)
    return "\n".join(lines[part_c_start:end_idx]).strip()

def extract_part_d_section(text):
    lines = text.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "PART C")
        part_d_start = None
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip().upper() == "PART D":
                part_d_start = j + 1
                break
        if part_d_start is None:
            return "PART D header not found after PART C."
    except StopIteration:
        return "PART C not found. Cannot extract PART D."
    end_idx = None
    for j in range(part_d_start, len(lines)):
        if lines[j].strip().upper() == "PART E":
            end_idx = j
            break
    end_idx = end_idx or len(lines)
    return "\n".join(lines[part_d_start:end_idx]).strip()

def extract_part_f_section(text):
    lines = text.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "PART E")
        part_f_start = None
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip().upper() == "PART F":
                part_f_start = j + 1
                break
        if part_f_start is None:
            return "PART F header not found after PART E."
    except StopIteration:
        return "PART E not found. Cannot extract PART F."
    end_idx = None
    for j in range(part_f_start, len(lines)):
        if lines[j].strip().upper() == "PART G":
            end_idx = j
            break
    end_idx = end_idx or len(lines)
    return "\n".join(lines[part_f_start:end_idx]).strip()

def extract_sections(text):
    lines = text.splitlines()
    sections = {}
    header_positions = []
    for i, line in enumerate(lines):
        if is_header_line(line):
            matched_header = line.strip().upper()
            header_positions.append((matched_header, i))
    for idx, (header, start_idx) in enumerate(header_positions):
        if header in ["FORWARDING LETTER", "SCHEDULE", "PART C", "PART D", "PART F"]:
            continue
        end_idx = header_positions[idx + 1][1] if idx + 1 < len(header_positions) else len(lines)
        content = "\n".join(lines[start_idx:end_idx]).strip()
        sections[header] = content
    return sections

def get_section_difference_with_gpt(section_name, filed_text, customer_text):
    client = get_azure_client()
    if not client:
        return "Error: Azure OpenAI client not configured. Please check your credentials."
    
    if not filed_text.strip() and not customer_text.strip():
        return "Section missing in both copies."
    if not filed_text.strip():
        return "Section present in Customer Copy but missing in Filed Copy."
    if not customer_text.strip():
        return "Section present in Filed Copy but missing in Customer Copy."

    prompt = f"""
You are a compliance analyst comparing the **{section_name}** section from two insurance policy documents: the Filed Copy and the Customer Copy.
 
---
 
### Your task:
Perform a **strict clause-by-clause or field-by-field comparison** between the two documents.
 
You must only report **meaningful structural or contextual differences**. Avoid any cosmetic, formatting, or placeholder-related differences.
 
---
 
### Mandatory Comparison Rules:
 
1. **Placeholder Handling (VERY STRICT):**
   - Do **not** flag a difference if the only variation is between a **placeholder** and a filled-in value.
   - Placeholders may appear as:
     - `<xx>`, `<xxxx>`, `<xxxxxx>`
     - `<dd-mm-yyyy>`, `<date>`
     - `<amount>`, `<rupees>`, `<annuity type>`, `<sum assured>`
     - `<name>`, `<address>`, `<mobile>`, `<email>`
     - Or any token enclosed in angle brackets `< >`
   - If both copies contain the same clause but one has a placeholder and the other a real value, **this must not be treated as a difference**.
   - Only flag a difference if the **entire clause** is missing in one copy and present in the other.
   - Fields where one side has `-`, `NA`, or `Not Applicable` and the other has a placeholder.
 
2. **Ignore the following completely:**
   - Placeholder or dummy values like `<xxxxxx>`, `Rs. 1,00,000`, `<dd-mm-yyyy>`, etc.
   - Differences in formatting (capitalization, punctuation, line breaks, spacing).
   - Differences in clause or field numbering (e.g., "1.2", "1.3").
   - Signature blocks, authorized signatories, seals.
   - Company addresses, disclaimers, office headers, footers, page numbers, version numbers.
 
---
 
###  Section-specific checks:
 
- In **FORWARDING LETTER**:
  - Compare `Policy Name`, `Plan Type`, `Document Type`, and other clauses.
  - For clauses like “Your Policy requires Premiums to be paid for X years”:
    - If both copies contain the clause (even with a placeholder in one), ignore any value difference.
    - If the clause is entirely missing in one copy, flag it.
 
- In **SCHEDULE**:
  - Explicitly check for the clause: `Due Dates of First Annuity Instalment`.
  - If it exists in one copy and is missing in the other, report it as a meaningful difference.
  - Ignore formatting, numbering, and placeholder value differences.
 
- In **PART G**:
  - For the subsection: `"Address & Contact Details of Ombudsman Centres"`:
    - Check if the subsection is present in both.
    - If present, compare only the **city names** (not full addresses).
    - If missing in one copy, report it clearly.
    - Ignore formatting, phone/email differences.
 
---
 
###  Output format:
 
If the sections are identical:
**Both copies are identical.**
 
Otherwise, report only meaningful differences using this format:
 
• In the Customer Copy, the field "XYZ" is missing, which is present in the Filed Copy.  
• In the Filed Copy, the field "XYZ" is missing, which is present in the Customer Copy.  
• In the Customer Copy, the clause "XYZ" differs from the Filed Copy.  
• In the Customer Copy, instead of "Phone Number & Mobile No", only "Phone Number" is present.  
• In the Customer Copy, the clause about "Premiums to be paid" is entirely missing, but present in the Filed Copy.  
 
---

### Comparison Inputs:

Filed Copy - {section_name}:
{filed_text}

Customer Copy - {section_name}:
{customer_text}
"""

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during comparison: {str(e)}"

st.title("PDF Policy Comparison Tool")

if azure_endpoint and azure_api_key:
    st.success("Azure OpenAI configured")
else:
    st.warning("Please configure Azure OpenAI credentials in the sidebar before proceeding.")

st.subheader("Upload Documents")
col1, col2 = st.columns(2)
with col1:
    filed_copy = st.file_uploader("Upload Filed Copy", type="pdf", key="filed")
with col2:
    customer_copy = st.file_uploader("Upload Customer Copy", type="pdf", key="customer")

if st.button("Extract and Download Excel", use_container_width=True):
    if not azure_endpoint or not azure_api_key:
        st.error("Please configure Azure OpenAI credentials in the sidebar first.")
    elif filed_copy and customer_copy:
        with st.spinner("Processing PDFs and analyzing sections..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Extracting text from Filed Copy...")
            progress_bar.progress(10)
            filed_text, _ = extract_final_filtered_pdf(filed_copy.read(), is_customer_copy=False)
            
            status_text.text("Extracting text from Customer Copy...")
            progress_bar.progress(20)
            customer_text, _ = extract_final_filtered_pdf(customer_copy.read(), is_customer_copy=True)

            status_text.text("Extracting product information...")
            progress_bar.progress(30)
            product_name, product_uin = extract_product_name_and_uin(filed_text)
            sample_id = extract_sample_number_from_filename(customer_copy.name)

            status_text.text("Extracting document sections...")
            progress_bar.progress(40)
            filed_sections = extract_sections(filed_text)
            customer_sections = extract_sections(customer_text)
            
            filed_sections["FORWARDING LETTER"] = extract_forwarding_letter_section_filed(filed_text)
            customer_sections["FORWARDING LETTER"] = extract_forwarding_letter_section_customer(customer_text)
            filed_sections["SCHEDULE"] = extract_schedule_section(filed_text)
            customer_sections["SCHEDULE"] = extract_schedule_section(customer_text)
            filed_sections["PART C"] = extract_part_c_section(filed_text)
            customer_sections["PART C"] = extract_part_c_section(customer_text)
            filed_sections["PART D"] = extract_part_d_section(filed_text)
            customer_sections["PART D"] = extract_part_d_section(customer_text)
            filed_sections["PART F"] = extract_part_f_section(filed_text)
            customer_sections["PART F"] = extract_part_f_section(customer_text)

            data = []
            total_sections = len(SECTION_HEADERS)
            
            for i, section in enumerate(SECTION_HEADERS):
                status_text.text(f"Analyzing {section} section with AI...")
                progress = 50 + (i * 40 / total_sections)
                progress_bar.progress(int(progress))
                
                filed_sec = filed_sections.get(section, "")
                cust_sec = customer_sections.get(section, "")
                difference = get_section_difference_with_gpt(section, filed_sec, cust_sec)
                data.append({
                    "Product Name": product_name,
                    "Product UIN": product_uin,
                    "Samples affected": sample_id,
                    "Observation - Category": "Mismatch between Filed Copy and Customer Copy",
                    "Page": section,
                    "Sub-category of Observation": difference
                })

            status_text.text("Generating Excel report...")
            progress_bar.progress(95)
            
            df = pd.DataFrame(data)
            
            st.success("Analysis completed!")
            st.subheader("Analysis Results")
            st.dataframe(df[["Product Name","Product UIN","Samples affected", "Observation - Category","Page","Sub-category of Observation"]], use_container_width=True)
                
            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="Observations", index=False, startrow=1, header=False)
                workbook = writer.book
                worksheet = writer.sheets["Observations"]
                
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'middle',
                    'align': 'center',
                    'fg_color': '#99d8f7',
                    'border': 1
                })
                merge_format = workbook.add_format({
                    'bold': True,
                    'align': 'center',
                    'valign': 'vcenter',
                    'text_wrap': True,
                    'border': 1
                })
                
                headers = list(df.columns)
                for col_num, header in enumerate(headers):
                    worksheet.write(0, col_num, header, header_format)
                
                grouped = df.groupby(["Product Name", "Product UIN", "Samples affected","Observation - Category"], sort=False)
                row_offset = 1
                for _, group in grouped:
                    group_len = len(group)
                    if group_len > 1:
                        worksheet.merge_range(row_offset, 0, row_offset + group_len - 1, 0, group.iloc[0]["Product Name"], merge_format)
                        worksheet.merge_range(row_offset, 1, row_offset + group_len - 1, 1, group.iloc[0]["Product UIN"], merge_format)
                        worksheet.merge_range(row_offset, 2, row_offset + group_len - 1, 2, group.iloc[0]["Samples affected"], merge_format)
                        worksheet.merge_range(row_offset, 3, row_offset + group_len - 1, 3, group.iloc[0]["Observation - Category"], merge_format)
                    row_offset += group_len

            progress_bar.progress(100)
            status_text.text("Ready for download!")
            output_excel.seek(0)
            
            st.download_button(
                label="Download Excel Report",
                data=output_excel.getvalue(),
                file_name=f"policy_comparison_{sample_id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            progress_bar.empty()
            status_text.empty()
    else:
        st.warning("⚠Please upload both PDF files.")
