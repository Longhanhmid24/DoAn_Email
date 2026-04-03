import tkinter as tk
from tkinter import ttk, messagebox
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from deep_translator import GoogleTranslator

# ==========================================
# LOAD MODEL (BASE + LORA)
# ==========================================
print("Loading TinyLlama...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 👉 base model (tự tải nếu chưa có)
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 👉 LoRA của bạn
lora_path = "/home/vophilong/Documents/Deep_learning/Models/TinyLLaMa/tinyllama_lora_tuned_best"

tokenizer = None
model = None

try:
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # load LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)

    # 👉 OPTIONAL: merge để chạy nhanh hơn
    model = model.merge_and_unload()

    print("Model loaded successfully")

except Exception as e:
    print("Model load error:", e)


# ==========================================
# GENERATE FUNCTION
# ==========================================
def generate_reply():
    global tokenizer, model

    if tokenizer is None or model is None:
        messagebox.showerror("Error", "Model not loaded")
        return

    context_text_vi = text_context.get("1.0", tk.END).strip()
    email_send_text_vi = text_email_send.get("1.0", tk.END).strip()

    if not email_send_text_vi:
        messagebox.showwarning("Warning", "Please enter email content")
        return

    try:
        lbl_status.config(text="Translating input...")
        root.update()

        translator = GoogleTranslator(source='vi', target='en')
        context_en = translator.translate(context_text_vi) if context_text_vi else ""
        email_en = translator.translate(email_send_text_vi)

        prompt = f"""
You are a professional customer support assistant.
Write a polite and helpful email reply.

Context:
{context_en}

Customer Email:
{email_en}

Reply:
"""

        lbl_status.config(text="Generating reply...")
        root.update()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Reply:" in generated:
            reply_en = generated.split("Reply:")[-1].strip()
        else:
            reply_en = generated

        lbl_status.config(text="Translating output...")
        root.update()

        translator = GoogleTranslator(source='en', target='vi')
        reply_vi = translator.translate(reply_en)

        text_email_reply.config(state=tk.NORMAL)
        text_email_reply.delete("1.0", tk.END)
        text_email_reply.insert(tk.END, reply_vi)
        text_email_reply.config(state=tk.DISABLED)

        lbl_status.config(text="Done")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        lbl_status.config(text="Error")


# ==========================================
# UI
# ==========================================
root = tk.Tk()
root.title("Auto-Reply Email AI - TinyLlama")
root.geometry("750x650")
root.configure(padx=20, pady=20)

style = ttk.Style()
style.configure("TLabel", font=("Arial", 11, "bold"))

ttk.Label(root, text="Context (optional):").pack(anchor="w")
text_context = tk.Text(root, height=4, width=80)
text_context.pack(pady=10)

ttk.Label(root, text="Customer Email:").pack(anchor="w")
text_email_send = tk.Text(root, height=6, width=80)
text_email_send.pack(pady=10)

tk.Button(root, text="Generate Reply", command=generate_reply).pack(pady=10)

ttk.Label(root, text="AI Reply:").pack(anchor="w")
text_email_reply = tk.Text(root, height=10, width=80, state=tk.DISABLED)
text_email_reply.pack(pady=10)

lbl_status = tk.Label(root, text="Ready")
lbl_status.pack(anchor="w")

root.mainloop()