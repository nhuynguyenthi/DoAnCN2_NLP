import streamlit as st
import tensorflow as tf
import pickle
import re
from keras.preprocessing.sequence import pad_sequences
from pyvi import ViTokenizer  # Xử lý tiếng Việt
from googletrans import Translator  # Thư viện dịch ngôn ngữ
import base64


# Hàm chuyển hình ảnh thành base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# Tải mô hình đã lưu
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(r'D:\Ung_dung_PTCX\model_cnn_bilstm.keras')
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {str(e)}")
        return None


# Tải tokenizer đã lưu
@st.cache_resource
def load_tokenizer():
    try:
        with open(r"D:\Ung_dung_PTCX\tokenizer_data.pkl", "rb") as input_file:
            tokenizer = pickle.load(input_file)
        return tokenizer
    except Exception as e:
        st.error(f"Lỗi khi tải tokenizer: {str(e)}")
        return None


def is_vietnamese(text):
    # Hàm kiểm tra xem văn bản có phải tiếng Việt không
    vietnamese_chars = "áàảãạâấầẩẫậăắằẳẵặđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    for char in vietnamese_chars:
        if char in text.lower():
            return True
    return False

def load_stopwords(filepath):
    # Hàm tải danh sách stopwords từ file
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            stopwords = set(f.read().splitlines())
        return stopwords
    except Exception as e:
        st.error(f"Lỗi khi tải stopwords: {str(e)}")
        return set()

def remove_stopwords(text, stopwords):
    # Loại bỏ ký tự đặc biệt, chỉ giữ lại chữ cái và số
    text_cleaned = re.sub(r'[^\w\s]', '', text)  # Giữ lại chữ cái, số và khoảng trắng
    # Tách văn bản thành danh sách từ
    words = text_cleaned.split()
    # Loại bỏ stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords]
    # Kết hợp lại thành văn bản
    return " ".join(filtered_words)

# Hàm tiền xử lý câu đầu vào
def preprocess_raw_input(raw_input, tokenizer):
    # Hàm tiền xử lý văn bản đầu vào
    input_text_pre = tf.keras.preprocessing.text.text_to_word_sequence(raw_input)
    input_text_pre = " ".join(input_text_pre)

    if is_vietnamese(input_text_pre):
        input_text_pre = ViTokenizer.tokenize(input_text_pre)

    # Tải và loại bỏ stopwords
    stopwords = load_stopwords(r"D:\Ung_dung_PTCX\stop_words.txt")
    input_text_pre = remove_stopwords(input_text_pre, stopwords)

    # Tokenize và pad sequences
    tokenized_data_text = tokenizer.texts_to_sequences([input_text_pre])
    vec_data = pad_sequences(tokenized_data_text, padding='post', maxlen=50)
    return vec_data


# Hàm dự đoán
def prediction(raw_input, tokenizer, model):
    input_model = preprocess_raw_input(raw_input, tokenizer)
    result, conf = inference_model(input_model, model)
    return result, conf


# Hàm infer
def inference_model(input_feature, model):
    output = model(input_feature).numpy()[0]
    result = output.argmax()
    conf = float(output.max())
    label_dict = {0: 'tiêu cực', 1: 'trung lập', 2: 'tích cực'}
    return label_dict[result], conf


# Hàm dịch ngôn ngữ
def translate_text(text, src_lang, dest_lang):
    translator = Translator()  # Tạo đối tượng Translator từ thư viện Google Translate
    try:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)  # Dịch văn bản
        if translated is None or not hasattr(translated, 'text'):  # Kiểm tra kết quả dịch
            raise ValueError("API Google Translate không trả về kết quả.")  # Nếu không có kết quả
        return translated.text  # Trả về văn bản dịch
    except Exception as e:
        return f"Lỗi khi dịch: {str(e)}"  # Bắt lỗi nếu có bất kỳ vấn đề nào

# Trang chính
def home_page():
    # CSS để tùy chỉnh giao diện
    st.markdown("""
        <style>
        .main-container {
            background-color: #f9f9f9;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .header {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .description {
            text-align: center;
            font-size: 18px;
            color: #34495e;
            margin-bottom: 30px;
        }
        .functions {
            margin-top: 10px;
            font-size: 18px;
            color: #555555;
        }
        .functions ul {
            list-style-type: none;
            padding-left: 0;
        }
        .functions li {
            margin: 15px 0;
            display: flex;
            align-items: center;
        }
        .functions li span.icon {
            font-size: 24px;
            margin-right: 10px;
        }
        img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 60%;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    # Hiển thị giao diện chính
    image_base64 = image_to_base64(r"D:\Ung_dung_PTCX\decoration.png")  # Sử dụng hình ảnh đã cập nhật
    st.markdown(f"""
    <div class="main-container">
        <div class="header">Ứng dụng phân tích cảm xúc đa ngôn ngữ</div>
        <div class="description">
            Phân tích cảm xúc từ văn bản một cách nhanh chóng và hiệu quả.
        </div>
        <img src="data:image/png;base64,{image_base64}" alt="sentiment image">
        <div class="functions">
            <h4>Chức năng chính:</h4>
            <ul>
                <li><span class="icon">💭</span><b>Phân tích cảm xúc:</b> Tiêu cực, Trung lập, Tích cực.</li>
                <li><span class="icon">🌐</span><b>Dịch ngôn ngữ:</b> Tự động dịch giữa Tiếng Việt và Tiếng Anh.</li>
                <li><span class="icon">📊</span><b>Tìm hiểu dữ liệu:</b> Thông tin chi tiết về tập dữ liệu.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Trang giới thiệu dữ liệu
def data_intro_page():
    # CSS tùy chỉnh
    st.markdown("""
        <style>
        .header-container {
            text-align: center;
            margin-bottom: 30px;
        }
        .header-container h1 {
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 5px;
        }
        .header-container p {
            font-size: 18px;
            color: #7f8c8d;
        }
        .data-section {
            margin-top: 20px;
            margin-bottom: 30px;
        }
        .data-section h4 {
            font-size: 24px;
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        .data-section ul {
            margin-top: 10px;
            list-style: none;
            padding-left: 0;
        }
        .data-section ul li {
            font-size: 18px;
            color: #2d3436;
            margin-bottom: 10px;
        }
        .data-section ul li span {
            font-weight: bold;
            color: #2980b9;
        }
        .data-section ul li span.negative {
            color: #e74c3c;
        }
        .data-section ul li span.neutral {
            color: #f39c12;
        }
        .data-section ul li span.positive {
            color: #27ae60;
        }
        </style>
    """, unsafe_allow_html=True)

    # Giao diện chính
    st.markdown("""
    <div class="header-container">
        <h1>📊 Giới thiệu Dữ liệu</h1>
        <p>Tìm hiểu tổng quan về các bộ dữ liệu được sử dụng để phân tích cảm xúc.</p>
    </div>
    """, unsafe_allow_html=True)

    # Dữ liệu tiếng Việt
    st.markdown("""
    <div class="data-section">
        <h4>📘 Dữ liệu Tiếng Việt</h4>
        <ul>
            <li><span>📍 Nguồn gốc:</span> Bộ dữ liệu UIT-VSFC (Vietnamese Students Feedback Corpus).</li>
            <li><span>📊 Số lượng:</span> 11,426 phản hồi.</li>
            <li><span>📋 Mô tả:</span> Gồm các câu phản hồi từ sinh viên, thuộc các chủ đề về chương trình đào tạo, cơ sở vật chất, giảng viên và khác.</li>
            <li><span>📌 Phân loại cảm xúc:</span> <span class="negative">❌ Tiêu cực</span>, <span class="neutral">➖ Trung lập</span>, <span class="positive">✅ Tích cực</span>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Dữ liệu tiếng Anh
    st.markdown("""
    <div class="data-section">
        <h4>📙 Dữ liệu Tiếng Anh</h4>
        <ul>
            <li><span>📍 Nguồn gốc:</span> Thu thập từ phản hồi khách hàng trong lĩnh vực nhà hàng.</li>
            <li><span>📊 Số lượng:</span> 3,693 câu phản hồi.</li>
            <li><span>📋 Mô tả:</span> Gồm các nhận xét đánh giá về chất lượng phục vụ, món ăn và không gian nhà hàng.</li>
            <li><span>📌 Phân loại cảm xúc:</span> <span class="negative">❌ Tiêu cực</span>, <span class="neutral">➖ Trung lập</span>, <span class="positive">✅ Tích cực</span>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Trang dự đoán
def prediction_page(model, tokenizer):
    st.title("🌟 Dự đoán cảm xúc 🌟")
    st.markdown("### Nhập văn bản và để AI giúp bạn phân tích cảm xúc!")

    # Thêm CSS để tùy chỉnh nút
    st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50; /* Màu nền nút */
            color: white; /* Màu chữ */
            padding: 10px 20px; /* Đệm */
            font-size: 16px; /* Kích thước chữ */
            border: none; /* Xóa viền */
            border-radius: 8px; /* Bo góc */
            cursor: pointer;
            transition-duration: 0.4s; /* Hiệu ứng hover */
        }
        .stButton>button:hover {
            background-color: #45a049; /* Màu khi hover */
        }
    </style>
    """, unsafe_allow_html=True)

    # Khu vực nhập văn bản
    text = st.text_area("✍️ Văn bản", "", height=150, placeholder="Ví dụ: Tôi cảm thấy rất vui hôm nay!")

    if text.strip():
        source_lang, dest_lang = ('vi', 'en') if is_vietnamese(text) else ('en', 'vi')

        # Tạo bố cục hàng ngang cho các nút
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌐 Dịch sang ngôn ngữ khác"):
                try:
                    translated_text = translate_text(text, src_lang=source_lang, dest_lang=dest_lang)
                    st.markdown(f"**Dịch ({'🇻🇳 Tiếng Việt' if source_lang == 'en' else '🇺🇸 Tiếng Anh'}):**")
                    st.info(translated_text)
                except Exception as e:
                    st.error(f"❌ Lỗi khi dịch: {str(e)}")

        with col2:
            if st.button("📊 Phân tích cảm xúc"):
                try:
                    result, conf = prediction(text, tokenizer, model)

                    # Hiển thị cảm xúc bằng hiệu ứng
                    if result == "tiêu cực":
                        st.snow()
                        st.image(r"D:\Ung_dung_PTCX\negative.png", caption="Cảm xúc Tiêu cực")
                    elif result == "trung lập":
                        st.image(r"D:\Ung_dung_PTCX\neutral.png", caption="Cảm xúc trung lập")
                    else:
                        st.balloons()
                        st.image(r"D:\Ung_dung_PTCX\positive.png", caption="Cảm xúc Tích cực")

                    # Hiển thị kết quả và độ chính xác
                    st.success(f"**Dự đoán cảm xúc:** {result.upper()}")
                    st.info(f"🎯 **Độ chính xác:** {conf:.2f}%")
                except Exception as e:
                    st.error(f"❌ Lỗi khi phân tích cảm xúc: {str(e)}")
    else:
        st.warning("⚠️ Vui lòng nhập văn bản để phân tích cảm xúc!")


# Main function
def main():
    st.sidebar.title("Bảng Dashboard")
    app_mode = st.sidebar.radio("Chọn Trang", ["Trang chủ", "Giới thiệu dữ liệu", "Dự đoán"])

    if app_mode == "Trang chủ":
        home_page()
    elif app_mode == "Giới thiệu dữ liệu":
        data_intro_page()
    elif app_mode == "Dự đoán":
        model = load_model()
        tokenizer = load_tokenizer()
        if not model or not tokenizer:
            st.error("Không thể tải mô hình hoặc tokenizer. Vui lòng kiểm tra lại.")
        else:
            prediction_page(model, tokenizer)


if __name__ == "__main__":
    main()
