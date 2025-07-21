import streamlit as st
import fitz  # PyMuPDF
import io
import pandas as pd
import os
from openai import AzureOpenAI

st.set_page_config(layout="wide")
load_dotenv()

EXCLUDE_SECTIONS = [
    "First Premium Receipt",
    "Your Renewal Page",
]

SECTION_HEADERS = [
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

# Azure OpenAI Configuration UI
st.sidebar.header("Azure OpenAI Configuration")

# Get Azure OpenAI credentials from UI or environment variables
azure_endpoint = st.sidebar.text_input(
    "Azure OpenAI Endpoint", 
    value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    help="Your Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/)"
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

# Initialize Azure OpenAI client
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
        st.sidebar.error(f"Error initializing Azure OpenAI client: {str(e)}")
        return None

def is_image_only_page(page):
    text = page.get_text("text").strip()
    return not text and (page.get_images(full=True) or page.get_drawings())

def extract_final_filtered_pdf(uploaded_file_bytes, is_customer_copy=False):
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

        if is_image_only_page(page):
            debug_logs += log + " Skipped (Image-only page)\n"
            continue

        if any(section.lower() in text_lower for section in EXCLUDE_SECTIONS):
            debug_logs += log + "Skipped (Excluded section)\n"
            continue

        for header in SECTION_HEADERS:
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

### Your task:
- Perform a **strict clause-by-clause or field-by-field comparison** between the two versions.
- **Ignore differences in clause numbers** (e.g., "15)" vs "16)") if the **clause title and content are the same**. Focus on **clause content**, not numbering.
- **Ignore placeholders**, formatting differences (punctuation, casing, spacing, line breaks), and standard footers.

---

### Specific comparison rules:

1. **Clause Matching:**
   - Match clauses based on **titles/headings**.
   - Treat semantically equivalent headings as the same, e.g., "email id", "Email Id", "E-Mail ID", "EMAILID".

2. **Ignore the following elements completely:**
   - Clause number mismatches (if content is same)
   - Placeholder values like `<xxxx>`, `<dd-mm-yyyy>`, `<amount>`, `Rs. 1,00,000`, etc.
   - Signature blocks, authorized signatories, seals
   - Company addresses, disclaimers, office registration details
   - Fields with values '-', 'Not Applicable' in one copy but placeholder in the other

3. **Section-specific checks:**
   - In **FORWARDING LETTER**, ensure `Policy Name` and `Plan Type` match,Document Type ,fields,clauses
   - In **SCHEDULE**, explicitly check :`Due Dates of First Annuity Instalment`
       

---

### Output format:

If the sections are structurally identical:
**Both copies are identical.**

Otherwise, report only meaningful structural or contextual differences using this format:

• In the Customer Copy, the field "XYZ" is missing, which is present in the Filed Copy.  
• In the Filed Copy, the field "XYZ" is missing, which is present in the Customer Copy.
• In the Customer Copy, Document "XYZ" is used instead of Document "ABC" the Filed Copy.
• In the Filed Copy, Document "XYZ" is present but missed in Customer Copy.
• In the Filed Copy, the clause "ABC" is missing, which is present in the Customer Copy.  
• In the Customer Copy, the Policy Name / Policy Type "XYZ" differs from the Filed Copy.  
• In the Customer Copy, the field "Due Dates of First Annuity Instalment" is present, but missing in the Filed Copy. 

• In the Customer Copy, instead of "Phone Number & Mobile No", the field "Toll-Free Number" is used.  
• In the Customer Copy, "Premiums to be paid for" is missing, but present in the Filed Copy.  

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

# ---------------- Streamlit UI ----------------
st.title("PDF Policy Comparison Tool")

# Check if Azure OpenAI is configured
if not azure_endpoint or not azure_api_key:
    st.warning("⚠️ Please configure Azure OpenAI credentials in the sidebar before proceeding.")

# File upload section
col1, col2 = st.columns(2)
with col1:
    filed_copy = st.file_uploader("Upload Filed Copy", type="pdf", key="filed")
with col2:
    customer_copy = st.file_uploader("Upload Customer Copy", type="pdf", key="customer")

if st.button("Process and Download"):
    if not azure_endpoint or not azure_api_key:
        st.error("Please configure Azure OpenAI credentials in the sidebar first.")
    elif filed_copy and customer_copy:
        with st.spinner("Processing PDFs..."):
            filed_text, filed_pdf = extract_final_filtered_pdf(filed_copy.read(), is_customer_copy=False)
            customer_text, customer_pdf = extract_final_filtered_pdf(customer_copy.read(), is_customer_copy=True)

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
            progress_bar = st.progress(0)
            total_sections = len(SECTION_HEADERS)
            
            for i, section in enumerate(SECTION_HEADERS):
                progress_bar.progress((i + 1) / total_sections)
                filed_sec = filed_sections.get(section, "")
                cust_sec = customer_sections.get(section, "")
                difference = get_section_difference_with_gpt(section, filed_sec, cust_sec)
                data.append({
                    "Section": section,
                    "Filed Copy": filed_sec,
                    "Customer Copy": cust_sec,
                    "Meaningful Difference": difference
                })

            df = pd.DataFrame(data)
            st.success("Processing completed!")
            st.dataframe(df[["Section","Filed Copy","Customer Copy", "Meaningful Difference"]], use_container_width=True)

            excel_out = io.BytesIO()
            with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Sections")
            excel_out.seek(0)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("Download Excel", data=excel_out.getvalue(),
                                   file_name="extracted_sections.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col2:
                if filed_pdf:
                    st.download_button("Filtered Filed Copy PDF", data=filed_pdf.getvalue(), file_name="filtered_filed_copy.pdf")
            with col3:
                if customer_pdf:
                    st.download_button("Filtered Customer Copy PDF", data=customer_pdf.getvalue(), file_name="filtered_customer_copy.pdf")
    else:
        st.warning("Please upload both PDFs.")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    ### How to use this tool:
    
    1. **Configure Azure OpenAI** in the sidebar:
       - Enter your Azure OpenAI endpoint URL
       - Enter your API key
       - Set the API version (default: 2024-02-15-preview)
       - Enter your deployment name (model name)
    
    2. **Upload PDFs**: Upload both the Filed Copy and Customer Copy PDF files
    
    3. **Process**: Click "Process and Download" to compare the documents
    
    4. **Download**: Get the comparison results in Excel format and filtered PDFs
    
    ### Environment Variables (Optional):
    You can also set these environment variables instead of entering credentials manually:
    - `AZURE_OPENAI_ENDPOINT`
    - `AZURE_OPENAI_API_KEY`
    - `AZURE_OPENAI_API_VERSION`
    - `AZURE_OPENAI_DEPLOYMENT_NAME`
    """)
