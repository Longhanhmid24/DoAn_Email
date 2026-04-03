import tkinter as tk
from tkinter import ttk, messagebox
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator

# ==========================================
# 1. KHỞI TẠO VÀ TẢI MÔ HÌNH AI T5
# ==========================================
print("Đang tải AI Model T5... Vui lòng đợi vài giây!")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ĐƯỜNG DẪN TỚI THƯ MỤC 'T5' CHỨA MODEL CỦA BẠN (Sửa lại cho đúng với máy bạn)
    # Ví dụ: "/home/vophilong/Documents/Deep_learning/DoAn_Email/T5"
    model_path = "/home/vophilong/Documents/Deep_learning/Models/T5_Now/T5" 
    
    # Load Tokenizer và Model từ thư mục local
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
    print("Tải hệ thống T5 thành công! Đang mở giao diện...")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")

# ==========================================
# 2. HÀM XỬ LÝ KHI BẤM NÚT "TẠO PHẢN HỒI"
# ==========================================
def generate_reply():
    # Lấy nội dung Tiếng Việt từ giao diện
    context_text_vi = text_context.get("1.0", tk.END).strip()
    email_send_text_vi = text_email_send.get("1.0", tk.END).strip()
    
    if not email_send_text_vi:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập nội dung Email (Email Send)!")
        return
        
    try:
        # --- BƯỚC A: Dịch từ Việt sang Anh ---
        lbl_status.config(text="Đang dịch đầu vào sang Tiếng Anh...", fg="blue")
        root.update()
        
        translator_vi_to_en = GoogleTranslator(source='vi', target='en')
        context_text_en = translator_vi_to_en.translate(context_text_vi) if context_text_vi else ""
        email_send_text_en = translator_vi_to_en.translate(email_send_text_vi)
        
        # Ghép chuỗi chuẩn form. 
       
        formatted_input = f"reply email: context: {context_text_en} email: {email_send_text_en}"
        
        formatted_input = f"""reply email professionally to a client:
context: {context_text_en}
email: {email_send_text_en}
"""
        
        # --- BƯỚC B: AI T5 Sinh câu trả lời ---
        lbl_status.config(text="AI (T5) đang tạo câu trả lời...", fg="blue")
        root.update()
        
        # Mã hóa đầu vào
        inputs = tokenizer(formatted_input, return_tensors="pt", max_length=1020, truncation=True).to(device)
        
        # Sinh văn bản
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.2,     # 👉 ép câu dài hơn
                repetition_penalty=1.1  # 👉 giảm lặp
            )
        
        # Giải mã kết quả
        predicted_reply_en = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # --- BƯỚC C: Dịch câu trả lời từ Anh về Việt ---
        lbl_status.config(text="Đang dịch câu trả lời sang Tiếng Việt...", fg="blue")
        root.update()
        
        translator_en_to_vi = GoogleTranslator(source='en', target='vi')
        predicted_reply_vi = translator_en_to_vi.translate(predicted_reply_en)
        
        # --- IN KẾT QUẢ RA GIAO DIỆN ---
        text_email_reply.config(state=tk.NORMAL)
        text_email_reply.delete("1.0", tk.END)
        text_email_reply.insert(tk.END, predicted_reply_vi)
        text_email_reply.config(state=tk.DISABLED)
        
        lbl_status.config(text="Hoàn tất việc sinh văn bản bằng T5!", fg="green")
        
    except Exception as e:
        messagebox.showerror("Lỗi", f"Có lỗi xảy ra:\n{e}")
        lbl_status.config(text="Lỗi hệ thống", fg="red")

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN TKINTER
# ==========================================
root = tk.Tk()
root.title("Auto-Reply Email AI - T5 Model (Hỗ trợ Tiếng Việt)")
root.geometry("750x650")
root.configure(padx=20, pady=20)

style = ttk.Style()
style.configure("TLabel", font=("Arial", 11, "bold"))

# Khu vực 1
lbl_context = ttk.Label(root, text="1. Ngữ cảnh trước đó (Gõ Tiếng Việt - Tùy chọn):")
lbl_context.pack(anchor="w", pady=(0, 5))
text_context = tk.Text(root, height=4, width=80, font=("Arial", 11), bg="#f0f8ff")
text_context.pack(pady=(0, 15))

# Khu vực 2
lbl_email_send = ttk.Label(root, text="2. Email khách hàng gửi đến (Gõ Tiếng Việt):")
lbl_email_send.pack(anchor="w", pady=(0, 5))
text_email_send = tk.Text(root, height=6, width=80, font=("Arial", 11), bg="#ffffff")
text_email_send.pack(pady=(0, 15))

# Nút bấm
btn_generate = tk.Button(root, text="✨ TẠO PHẢN HỒI (AI T5 + GOOGLE TRANSLATE) ✨", bg="#4CAF50", fg="white", 
                         font=("Arial", 12, "bold"), command=generate_reply)
btn_generate.pack(pady=10)

# Khu vực 3
lbl_email_reply = ttk.Label(root, text="3. AI Đề Xuất Phản Hồi (Đã dịch sang Tiếng Việt):")
lbl_email_reply.pack(anchor="w", pady=(0, 5))
text_email_reply = tk.Text(root, height=10, width=80, font=("Arial", 11), bg="#e8f5e9", state=tk.DISABLED)
text_email_reply.pack(pady=(0, 5))

lbl_status = tk.Label(root, text="Sẵn sàng.", font=("Arial", 10), fg="gray")
lbl_status.pack(anchor="w", pady=5)

root.mainloop()