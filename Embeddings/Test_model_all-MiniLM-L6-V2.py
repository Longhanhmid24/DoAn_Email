import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. KHỞI TẠO VÀ TẢI MÔ HÌNH AI
# ==========================================
print("Đang tải AI Model và Cơ sở dữ liệu... Vui lòng đợi vài giây!")
try:
    # Tải model nhúng
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2').to(device) # Sử dụng mô hình all-MiniLM-L6-v2 để tạo vector embeddings
    
    # Tải Vector Database đã lưu ở Bước 8
    with open("/home/vophilong/Documents/Deep_learning/DoAn_Email/Embeddings/retrieval_system.pkl", "rb") as f:
        saved_data = pickle.load(f)
        train_embeddings = saved_data['embeddings']
        train_df = saved_data['dataframe']
    print("Tải hệ thống thành công! Đang mở giao diện...")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    print("Hãy chắc chắn bạn đã chạy xong Bước 8 và có file 'retrieval_system.pkl' ở cùng thư mục.")

# ==========================================
# 2. HÀM XỬ LÝ KHI BẤM NÚT "TẠO PHẢN HỒI"
# ==========================================
def generate_reply():
    # Lấy nội dung từ giao diện
    context_text = text_context.get("1.0", tk.END).strip()
    email_send_text = text_email_send.get("1.0", tk.END).strip()
    
    if not email_send_text:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập nội dung Email (Email Send)!")
        return
        
    # Ghép chuỗi theo đúng format của dataset: "context: ... | email: ..."
    formatted_input = f"context: {context_text} | email: {email_send_text}"
    
    # Hiển thị trạng thái đang xử lý
    lbl_status.config(text="AI đang suy nghĩ...", fg="blue")
    root.update()
    
    try:
        # AI xử lý
        input_vector = retriever_model.encode([formatted_input], convert_to_tensor=True)
        cos_scores = cosine_similarity(input_vector.cpu().numpy(), train_embeddings.numpy())
        
        best_idx = np.argmax(cos_scores)
        best_score = cos_scores[0][best_idx]
        predicted_reply = train_df['Output_Text'].iloc[best_idx]
        
        # In kết quả ra màn hình GUI
        text_email_reply.config(state=tk.NORMAL) # Mở khóa ô text để ghi
        text_email_reply.delete("1.0", tk.END)
        text_email_reply.insert(tk.END, predicted_reply)
        text_email_reply.config(state=tk.DISABLED) # Khóa lại không cho người dùng sửa
        
        lbl_status.config(text=f"Hoàn tất! Độ tự tin (Similarity Score): {best_score:.4f}", fg="green")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Có lỗi xảy ra trong quá trình sinh văn bản:\n{e}")
        lbl_status.config(text="Lỗi hệ thống", fg="red")

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN TKINTER
# ==========================================
root = tk.Tk()
root.title("Hệ Thống Tự Động Phản Hồi Email (AI Retrieval)")
root.geometry("700x650")
root.configure(padx=20, pady=20)

# Style
style = ttk.Style()
style.configure("TLabel", font=("Arial", 11, "bold"))
style.configure("TButton", font=("Arial", 12, "bold"), padding=5)

# --- KHU VỰC 1: INPUT NGỮ CẢNH (CONTEXT) ---
lbl_context = ttk.Label(root, text="1. Ngữ cảnh trước đó (Context - Tùy chọn):")
lbl_context.pack(anchor="w", pady=(0, 5))
text_context = tk.Text(root, height=4, width=80, font=("Arial", 11), bg="#f0f8ff")
text_context.pack(pady=(0, 15))

# --- KHU VỰC 2: INPUT EMAIL GỬI ĐẾN (EMAIL SEND) ---
lbl_email_send = ttk.Label(root, text="2. Email khách hàng gửi đến (Email Send):")
lbl_email_send.pack(anchor="w", pady=(0, 5))
text_email_send = tk.Text(root, height=6, width=80, font=("Arial", 11), bg="#ffffff")
text_email_send.pack(pady=(0, 15))

# --- NÚT BẤM SINH VĂN BẢN ---
btn_generate = tk.Button(root, text="✨ TẠO EMAIL PHẢN HỒI ✨", bg="#4CAF50", fg="white", 
                         font=("Arial", 12, "bold"), command=generate_reply)
btn_generate.pack(pady=10)

# --- KHU VỰC 3: OUTPUT AI (EMAIL REPLY) ---
lbl_email_reply = ttk.Label(root, text="3. AI Đề Xuất Phản Hồi (Email Reply):")
lbl_email_reply.pack(anchor="w", pady=(0, 5))
text_email_reply = tk.Text(root, height=8, width=80, font=("Arial", 11, "italic"), bg="#e8f5e9", state=tk.DISABLED)
text_email_reply.pack(pady=(0, 5))

# --- THANH TRẠNG THÁI ---
lbl_status = tk.Label(root, text="Sẵn sàng.", font=("Arial", 10), fg="gray")
lbl_status.pack(anchor="w", pady=5)

# Chạy vòng lặp ứng dụng
root.mainloop()