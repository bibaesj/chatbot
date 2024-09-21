import os
import PyPDF2

# PDF 파일에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        # PDF 파일 열기
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            # 모든 페이지에서 텍스트 추출
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# 텍스트를 .txt 파일로 저장
def save_text_to_file(text, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        print(f"Text successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving text to {output_path}: {e}")

# 메인 함수
def main():
    # PDF 파일 경로 설정
    pdf_directory = '/Users/macbook/Desktop/streamlit'
    
    # 디렉토리 내 모든 .pdf 파일 찾기
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            # 텍스트 추출
            extracted_text = extract_text_from_pdf(pdf_path)
            
            # 출력 파일 경로 설정 (.txt 파일로 저장)
            output_path = os.path.join(pdf_directory, filename.replace('.pdf', '.txt'))
            
            # 텍스트 파일로 저장
            save_text_to_file(extracted_text, output_path)

if __name__ == "__main__":
    main()
