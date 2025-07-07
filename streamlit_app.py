import streamlit as st
import fitz  # PyMuPDF
import io
import pandas as pd
import os
#from dotenv import load_dotenv
 
st.set_page_config(layout="wide")
load_dotenv()
 
EXCLUDE_SECTIONS = [
    "First Premium Receipt",
    "Your Renewal Page"
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
 
# ---------------- Streamlit UI ----------------
 
col1, col2 = st.columns(2)
with col1:
    filed_copy = st.file_uploader("Upload Filed Copy", type="pdf", key="filed")
with col2:
    customer_copy = st.file_uploader("Upload Customer Copy", type="pdf", key="customer")
 
if st.button("Process and Download"):
    if filed_copy and customer_copy:
        filed_text, filed_pdf = extract_final_filtered_pdf(filed_copy.read(), is_customer_copy=False)
        customer_text, customer_pdf = extract_final_filtered_pdf(customer_copy.read(), is_customer_copy=True)
 
        def is_header_line(line):
            return line.strip().upper() in SECTION_HEADERS
 
        def extract_forwarding_letter_section_filed(text):
            lines = text.splitlines()
            try:
                start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "FORWARDING LETTER")
                end_idx = next(i for i, line in enumerate(lines[start_idx+1:], start=start_idx+1)
                               if is_header_line(line) and line.strip().upper() != "FORWARDING LETTER")
                return "\n".join(lines[start_idx:end_idx]).strip()
            except StopIteration:
                return "FORWARDING LETTER section not found in Filed Copy."
 
        def extract_forwarding_letter_section_customer(text):
            lines = text.splitlines()
            try:
                end_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "PREAMBLE")
                return "\n".join(lines[:end_idx]).strip()
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
                if "GRIEVANCE REDRESSAL MECHANISM" in line_upper:
                    end_idx = j
                    break
                if line_upper in SECTION_HEADERS and line_upper != "SCHEDULE":
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
        for section in SECTION_HEADERS:
            data.append({
                "Section": section,
                "Filed Copy": filed_sections.get(section, ""),
                "Customer Copy": customer_sections.get(section, "")
            })
 
        df = pd.DataFrame(data)
 
        st.dataframe(df, use_container_width=True)
 
        excel_out = io.BytesIO()
        with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sections")
        excel_out.seek(0)
 
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download Excel", data=excel_out.getvalue(),
                               file_name="extracted_sections.xlsx", mime="application/vnd.openxmlf-ormats-officedocument.spreadsheetml.sheet")
        with col2:
            if filed_pdf:
                st.download_button("Filtered Filed Copy PDF", data=filed_pdf.getvalue(), file_name="filtered_filed_copy.pdf")
        with col3:
            if customer_pdf:
                st.download_button("Filtered Customer Copy PDF", data=customer_pdf.getvalue(), file_name="filtered_customer_copy.pdf")
    else:
        st.warning("Please upload both PDFs.")
