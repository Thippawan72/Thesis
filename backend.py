import streamlit as st
import os
from pythainlp.tokenize import Tokenizer, word_tokenize
from pythainlp.corpus.common import thai_stopwords, thai_words

# กำหนดชุดคำสำคัญ (important_words)
important_words = {
    "นี่", "หนังสือ", "นั่น", "สมุด", "อะไร", "นาฬิกา", "หรือ", "ปากกา", "คุณ", "เข้าใจ",
    "ฉัน", "ขอโทษ", "ขอบคุณ", "โทรศัพท์", "อยู่", "ที่ไหน", "คน", "ขาย", "ราคา", "บน",
    "โต๊ะ", "ญี่ปุ่น", "เกาหลีใต้", "ภาษาไทย", "ยาก", "ง่าย", "แพง", "เก้าอี้", "ชอบ",
    "อาหาร", "จีน", "ไทย", "อ่าน", "เขียน", "ไป", "ที่ไหน", "เขา", "ดื่ม", "ทำงาน",
    "พูด", "ช้า", "ลง", "อีก", "มี", "เวลา", "ว่าง", "ทุกวัน", "เสาร์",
    "เมื่อวานนี้", "หนึ่งสัปดาห์หน้า", "เรียน", "หนึ่งสัปดาห์ที่แล้ว", "กลับบ้าน", "สาม",
    "ปี", "กลับ", "ประเทศไทย", "ตอนเที่ยง", "กิน", "ข้าวมันไก่", "แล้ว", "ผม", "พบ",
    "แฟน", "เริ่ม", "ตอนนี้", "กี่โมง", "ข้าว", "เช้า", "สี่", "สนามบิน", "พี่ชาย",
    "สวนสัตว์", "เที่ยง", "ทำ", "ได้", "สามารถ", "ห้องน้ำ", "นาที",
    "เข้า", "ห้า", "ชั่วโมง", "สิบ", "กลับบ้าน", "ดิฉัน", "พรุ่งนี้", "ธนาคาร", "วันนี้",
    "วันหยุด", "ดี", "วันอาทิตย์", "สิ้นเดือน", "เงินเดือน", "ออก", "วันปีใหม่", "ไปเที่ยว",
    "เดือน", "นาน", "เมื่อวานซืน", "คุย", "วันเสาร์", "เดิน", "เล่น", "อังกฤษ", "ธันวาคม",
    "ออกกำลังกาย", "ยัง", "เมื่อไร", "อยาก", "เคย", "เม็กซิโก", "อากาศ", "ร้อน",
    "ฤดู", "หนาว", "พ่อ", "หา", "มาก", "ขับรถ", "รถยนต์", "ไทย", "ไม่มี", "หิมะ",
    "ตก", "ภูเขา", "เดี๋ยวนี้", "ต้นไม้", "บ้าน", "ธุระ", "ไม่", "ไม่กิน", "นั่ง",
    "ฟัง", "ร้องเพลงคาราโอเกะ", "ยืน", "ใคร", "ว่ายน้ำ", "ออกกำลังกาย", "จำ",
    "ได้", "ได้ยิน", "มอง", "เห็น", "รู้จัก", "เสร็จแล้ว", "นอนหลับ", "ไม่ดี",
    "ง่วงนอน", "ดื่ม", "กาแฟ", "ต้องการ", "เผ็ด", "คิดถึง", "มองดู", "ทะเล",
    "นิดหน่อย", "ภาษาอังกฤษ", "ผิด", "อร่อย", "ชื่อเล่น", "ตัด", "ผม (เส้นผม)",
    "โกนหนวด", "เช้า", "อาบน้ำ", "ซื้อ", "สบู่", "ยาสีฟัน", "เสื้อแขนยาว",
    "ครอบครัว", "แต่งงาน", "ลูก", "อายุ", "แมว", "ครู", "ทำไม", "ถ้า", "รวย",
    "ฝนตก", "หมอ", "ฟัน", "ลุง", "รัก", "สูง", "เกลียด", "โกหก", "สวย",
    "กล้วยหอม", "สอง", "สาม", "คอมพิวเตอร์", "หนึ่ง", "กรุงเทพฯ", "นั่ง",
    "ลง", "ใหม่", "ดอกไม้", "เท่าไร"
}

def get_lemma(word):
    lemmas = {
        "มาช้า": "มาสาย",
        "ขับ": "ขับรถ",
        "ทาน": "กิน",
        "ไม่ทาน": "ไม่กิน",
        "ผม ": "ผม (เส้นผม)",
        # สามารถเพิ่ม lemma ได้ตามต้องการ
    }
    return lemmas.get(word, word)

def reorder_to_tsl(words):
    # ลบคำที่ไม่จำเป็นออก
    simplified_sentence = [word for word in words if word not in ['คือ', 'เป็น', 'จะ', 'ก็', 'ที่', 'นั้น', 'นี้', 'ได้']]
    # คำที่เกี่ยวข้องกับเวลา
    time_words = ['วันนี้', 'พรุ่งนี้', 'เมื่อวาน', 'ตอนเช้า', 'ตอนเย็น', 'หนึ่งสัปดาห์หน้า']
    time_elements = [word for word in simplified_sentence if word in time_words]
    non_time_elements = [word for word in simplified_sentence if word not in time_words]
    # นำคำที่เกี่ยวข้องกับเวลาไปไว้ก่อน
    reordered_sentence = time_elements + non_time_elements
    return reordered_sentence

def process_sentence(sentence):
    # กำหนดคำสำหรับ tokenizer
    custom_words = set(thai_words())
    custom_words.discard("คนญี่ปุ่น")
    custom_words.discard("ราคาแพง")
    custom_words.discard("อาหารจีน")
    custom_words.discard("เวลาว่าง")
    custom_words.discard("กินข้าว")
    custom_words.discard("อ่านหนังสือ")
    custom_words.discard("เที่ยงตรง")
    custom_words.discard("ทำอาหาร")
    custom_words.discard("แต่เช้า")
    custom_words.discard("เดินเล่น")
    custom_words.discard("ฤดูหนาว")
    custom_words.discard("คุณพ่อ")
    custom_words.discard("ทานข้าว")
    custom_words.discard("มองเห็น")
    custom_words.discard("ตัดผม")
    custom_words.discard("คนไทย")
    custom_words.discard("ที่อยู่")
    custom_words.discard("หรือยัง")
    custom_words.discard("มีครอบครัว")
    custom_words.discard("เขียนหนังสือ")
    custom_words.discard("ยืดตัว")
    custom_words.discard("ไม้ดอก")
    custom_words.discard("")
    
    custom_words.add("เมื่อวานนี้")
    custom_words.add("หนึ่งสัปดาห์หน้า")
    custom_words.add("หนึ่งสัปดาห์ที่แล้ว")
    custom_words.add("ประเทศไทย")
    custom_words.add("กี่โมง")
    custom_words.add("มาช้า")
    custom_words.add("วันปีใหม่")
    custom_words.add("วันเสาร์")
    custom_words.add("ฝนตกหนัก")
    custom_words.add("ไม่มี")
    custom_words.add("ร้องเพลงคาราโอเกะ")
    custom_words.add("ไม่ทาน")
    custom_words.add("เสร็จแล้ว")
    custom_words.add("ไม่ดี")
    custom_words.add("โกนหนวด")
    custom_words.add("สีขาว")
    custom_words.add("สีดำ")
    custom_words.add("เสื้อยืด")
    
    stopwords = set(thai_stopwords())
    stopwords.add("ไหม")
    stopwords.add("กี่")
    stopwords.add(" ")
    stopwords.discard("อยู่")
    stopwords.discard("ตรง")
    stopwords.discard("มา")
    stopwords.discard("หรือยัง")
    stopwords.discard("ยัง")
    stopwords.discard("ได้")
    stopwords.discard("ใหญ่")
    stopwords.discard("ตัว")
    
    custom_tokenizer = Tokenizer(custom_words)
    tokens = custom_tokenizer.word_tokenize(sentence)
    
    # กรอง stopwords (ยกเว้น important_words)
    filtered_tokens = [token for token in tokens if token not in stopwords or token in important_words]
    # ทำ lemmatization
    lemmatized_tokens = [get_lemma(token) for token in filtered_tokens]
    # จัดเรียงคำใหม่
    reordered_sentence = reorder_to_tsl(lemmatized_tokens)
    
    return reordered_sentence

# ฟังก์ชันโหลดคีย์เวิร์ดและไฟล์วิดีโอ
video_directory = r"D:\Project_TSL\all_tsl_video"
keyword_file_path = r"D:\Project_TSL\keyword_video.txt"

def load_keyword_video_mapping(file_path):
    keyword_video_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # รูปแบบในไฟล์: video_filename - keyword
            video_file, keyword = line.strip().split(' - ')
            keyword_video_map[keyword] = video_file
    return keyword_video_map

def display_videos_inline(keywords, keyword_video_map):
    with st.container():
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown('<div class="results-title">ผลลัพธ์วิดีโอภาษามือไทย</div>', unsafe_allow_html=True)

        # Define the number of columns
        num_columns = 7  # Number of columns per row
        cols = st.columns(num_columns)  # Create columns

        for i, keyword in enumerate(keywords):
            col = cols[i % num_columns]  # Get the current column based on the index
            with col:
                if keyword in keyword_video_map:
                    video_file = keyword_video_map[keyword]
                    video_path = os.path.join(video_directory, video_file)

                    if os.path.exists(video_path):
                        st.write(f"ภาษามือของคำว่า **'{keyword}'**")
                        col.video(video_path, start_time=0, format="video/mp4", loop=True, autoplay=True)
                    else:
                        st.write(f"ไม่พบไฟล์วิดีโอ '{video_file}' สำหรับคำว่า **'{keyword}'**")
                else:
                    st.write(f"ภาษามือสำหรับคำว่า **'{keyword}'**")
                    empty_box = st.empty()  # Create an empty box
                    empty_box.markdown("""
                            <div style="border: 1px solid black; background-color: black; width: 185px; height: 185px; display: flex; justify-content: center; align-items: center;">
        <p style="color: white; text-align: center;"> No Result Found </div>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# โหลด mapping จากไฟล์วิดีโอ
keyword_video_map = load_keyword_video_mapping(keyword_file_path)

# กำหนดค่าเริ่มต้นให้เป็น Wide mode
st.set_page_config(layout="wide")

#ซ่อนแถบ Developer Options
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

st.markdown(hide_menu_style, unsafe_allow_html=True)

#เปลี่ยนสีแถบ Developer Options
change_menu_style = """
    <style>
    #MainMenu {background-color: #4CAF50;}  /* เปลี่ยนสีของแถบด้านบน */
    footer {background-color: #4CAF50;}  /* เปลี่ยนสีของ footer */
    </style>
    """
st.markdown(change_menu_style, unsafe_allow_html=True)

# สร้าง CSS สำหรับปุ่มลิ้งค์
about_button_style = """
    <style>
    .about-button { 
        font-family: 'Sarabun', sans-serif;
        font-weight: 700;
        padding: 10px 20px;
        background-color: #b9eaec; /* สีพื้นหลังของปุ่ม */
        color: white; /* สีข้อความภายในปุ่ม */
        border: none; /* ไม่มีกรอบ */
        border-radius: 5px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        position: absolute;
        top: 15px; /* ปุ่มจะอยู่ที่ด้านบน */
        right: 20px; /* ปุ่มจะอยู่ที่ด้านขวา */
    }
    
    .about-button:hover {
       color: white; /* สีข้อความภายในปุ่ม */
    }
    </style>
    """
# ใช้ CSS ผ่าน st.markdown เพื่อเพิ่มสไตล์ให้กับปุ่ม
st.markdown(about_button_style, unsafe_allow_html=True)

# ลิ้งค์ไปยังหน้า "เกี่ยวกับเรา"
about_url = "https://yourwebsite.com/about"  # URL ของหน้า "เกี่ยวกับเรา"
logo_path = "D:\\Project_TSL\\สถิติ.png"

# CSS สำหรับแถบหัวข้อที่ติดอยู่ด้านบนสุด
import base64
from PIL import Image
import io

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        image.thumbnail((80, 80))  # Resize the image to 50x50 pixels
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

header_style = f"""
    <style>
     @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;700&display=swap'); /* นำเข้าฟอนต์ Sarabun จาก Google Fonts */
    .header {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #b9eaec; /* สีฟ้า */
        color: black;
        padding: 40px;
        display: flex; /* ใช้ flex เพื่อจัดการให้อยู่ตรงกลาง */
        align-items: center; /* จัดให้อยู่ตรงกลางแนวตั้ง */
    }}  
     .logo img {{
        width: 80px;  /* ขนาดรูปภาพตรา */
        margin-right: 20px;  /* ระยะห่างระหว่างรูปภาพและข้อความ */
    }}
    .header-title {{
        font-size: 24px;
        font-weight: bold;
        font-family: 'Sarabun', sans-serif; /* เปลี่ยนฟอนต์ของ title เป็น Sarabun */
    }}
    .header-subtitle {{
        font-size: 18px;
        font-weight: 300;
        font-family: 'Sarabun', sans-serif; /* เปลี่ยนฟอนต์ของ subtitle เป็น Sarabun */
    }}
    .header-button {{
        margin-left: auto; /* จัดปุ่มไปที่ด้านขวา */
    }}
    .content {{
        padding-top: 20px; /* เลื่อนเนื้อหาลงมาเพื่อไม่ให้ทับกับ header */
    }}
    </style>
    <div class="header">
        <div class="logo">
            <img src="data:image/png;base64,{get_image_base64(logo_path)}" alt="Logo">
        </div>
        <div>
            <div class="header-title">เว็บแอพพลิเคชันแปลข้อความเป็นวิดีโอภาษามือไทยสำหรับผู้เริ่มต้นเรียนรู้ภาษามือไทย</div>
            <div class="header-subtitle">Thai Text-to-Sign Language Video Translation Website for Beginner</div>
        </div>
        <div class="header-button">
            <a href="{about_url}" target="_blank" class="about-button">เกี่ยวกับเรา</a>
        </div>
    </div>
    <div class="content">
    </div>
    """
    
 # ใช้ CSS ผ่าน st.markdown เพื่อแสดงส่วนหัวที่ติดอยู่ด้านบนสุด
st.markdown(header_style, unsafe_allow_html=True)

# เพิ่มพื้นที่ให้เนื้อหาหลักเพื่อไม่ให้ทับกับ header
st.markdown('<div class="content"></div>', unsafe_allow_html=True)

# ตรวจสอบว่ามีการกำหนดค่าเริ่มต้นของ text_input_value ใน session_state หรือไม่
if 'text_input_value' not in st.session_state:
    st.session_state['text_input_value'] = ''  # กำหนดค่าเริ่มต้นเป็นสตริงว่าง

# ฟังก์ชันสำหรับลบข้อความ
def clear_text():
    st.session_state['text_input_value'] = ''  # ล้างค่ากลับเป็นสตริงว่าง

# CSS สำหรับกล่องป้อนข้อความและปุ่ม
input_box_style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap'); /* นำเข้าฟอนต์ Sarabun */
    
    /* สไตล์สำหรับกล่อง input */
    div[data-baseweb="input"] > input {
        font-size: 18px;
        padding: 20px;
        border-radius: 8px;
        border: 2px solid #4CAF50; /* สีขอบ */
        background-color: #f0f8ff; /* สีพื้นหลัง */
        color: #333333;
        width: 100%;
        box-sizing: border-box;
        font-family: 'Sarabun', sans-serif; /* กำหนดฟอนต์ Sarabun */
    }

    /* สไตล์สำหรับ label (ข้อความที่บอกว่า "ป้อนข้อความที่ต้องการแปลงเป็นภาษามือ") */
     input::label {
        font-family: 'Sarabun', sans-serif; /* กำหนดฟอนต์ให้กับ label */
        font-size: 40px; /* ขนาดฟอนต์ */
        color: #333333; /* สีข้อความ */
    }

    /* สไตล์สำหรับ placeholder */
    input::placeholder {
        color: #888888; /* สีของ placeholder */
        font-size: 16px;
        font-family: 'Sarabun', sans-serif; /* กำหนดฟอนต์ Sarabun สำหรับ placeholder */
    }

    /* สไตล์สำหรับปุ่ม */
    .streamlit-button {
        font-family: 'Sarabun', sans-serif; /* ฟอนต์ Sarabun */
        font-size: 16px; /* ขนาดฟอนต์ */
        font-weight: 700; /* ตัวหนา */
        padding: 10px 20px; /* ระยะห่างภายในปุ่ม */
        background-color: #4CAF50; /* สีพื้นหลังของปุ่ม */
        color: white; /* สีข้อความภายในปุ่ม */
        border: none; /* ไม่มีกรอบ */
        border-radius: 5px; /* มุมโค้งมน */
        cursor: pointer; /* เปลี่ยนเคอร์เซอร์เมื่อชี้ที่ปุ่ม */
    }
    
    /* สไตล์เมื่อชี้ไปที่ปุ่ม */
    .streamlit-button:hover {
        color: white; /* สีข้อความภายในปุ่ม */
    }
    </style>
    """

# ใช้ CSS ผ่าน st.markdown เพื่อเพิ่มสไตล์ให้กับกล่องป้อนข้อความและปุ่ม
st.markdown(input_box_style, unsafe_allow_html=True)

# กำหนด session state สำหรับกล่องข้อความ
if 'text_input_value' not in st.session_state:
    st.session_state['text_input_value'] = ''

# ฟังก์ชันสำหรับลบข้อความ
def clear_text():
    st.session_state['text_input_value'] = ''

# สร้างคอลัมน์เพื่อจัดให้อยู่ตรงกลาง
col1, col2, col3 = st.columns([1, 2, 1])  # col2 จะเป็นคอลัมน์กลางและใหญ่กว่า

# สร้างตัวแปร result ที่เริ่มต้นเป็น None
result = None

with col2:  # ใช้คอลัมน์กลางสำหรับกล่องและปุ่ม
    # สร้างกล่องข้อความ
    sentence_input = st.text_input("ป้อนข้อความที่ต้องการแปลงเป็นภาษามือ", 
                                   st.session_state['text_input_value'], 
                                   key='text_input_box', 
                                   placeholder="พิมพ์ข้อความที่นี่...",)

    # สร้างปุ่มส่งข้อความ
    if st.button("ส่งข้อความ"):
        if sentence_input:
            result = process_sentence(sentence_input)  # กำหนดค่าให้กับ result

            # ลบข้อความหลังจากการส่งข้อความ
            st.session_state['text_input_value'] = ''
        else:
            st.warning("กรุณาป้อนข้อความ")

# หาก result มีค่า ให้แสดงวิดีโอ
if result:
    display_videos_inline(result, keyword_video_map)


# กำหนด CSS สำหรับ footer
footer_style = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: left;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>ติดต่อเรา 📞 020-0220-200 ✉️ TSLTranslation@gmail.com</p>
    </div>
    """

# ใช้ CSS ผ่าน st.markdown
st.markdown(footer_style, unsafe_allow_html=True)
