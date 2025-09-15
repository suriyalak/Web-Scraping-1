import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import time


st.set_page_config(
    page_title="Dallas Mavericks",
    layout="centered",
    initial_sidebar_state="expanded"
)


## team name
title_cols = st.columns(3)
with title_cols[0]:
    st.image('dallasmavericks.png', caption='Dallas Mavericks')
    
with title_cols[1]:
    st.write('  ') 
    st.title('Dallas Mavericks')


st.markdown(
    """
    ##### About team
    The Dallas Mavericks are an American professional basketball team based in Dallas.
    The Mavericks compete in the NBA as a member of the Southwest Division of the Western Conference.
    The team plays its home games at American Airlines Center, which it shares with the National Hockey League's Dallas Stars.
    """
)





## team players
st.header('⛹🏻 Team Players')

with st.expander("Show the Roster"):

    players_cols1 = st.columns(5)
    with players_cols1[0]:
        st.image('max.jpg')
        st.write('Max Christie')
        st.write('00')

        st.image('jeden.jpg')
        st.write('Jaden Hardy')
        st.write('1')
        
        st.image('dwight.jpg')
        st.write('Dwight Powell')
        st.write('7')
        
        
    with players_cols1[1]:
        st.image('anthony.jpg')
        st.write('Anthony Davis')
        st.write('3')
        
        st.image('kyrie.jpg')
        st.write('Kyrie Irving')
        st.write('11')
        
        st.image('dangelo.jpg')
        st.write('D Angelo Russell')
        st.write('15')
        
        
        
    with players_cols1[2]:
        st.image('dante.jpg')
        st.write('Danté Exum')
        st.write('0')

        st.image('dereck.jpg')
        st.write('Dereck Lively II')
        st.write('2')
        
        st.image('klay.jpg')
        st.write('Klay Thompson')
        st.write('31')
        
    with players_cols1[3]:
        st.image('cooper.jpg')
        st.write('Cooper Flagg')
        st.write('32')
        
        st.image('naji.jpg')
        st.write('Naji Marshall')
        st.write('13')
        
        st.image('pjwash.jpg')
        st.write('P. J. Washington')
        st.write('25')
        

    with players_cols1[4]:
        st.image('daniel.jpg')
        st.write('Daniel Gafford')
        st.write('21')
        
        st.image('caleb.jpg')
        st.write('Caleb Martin')
        st.write('16')
        
        st.image('brandon.jpg')
        st.write('Brandon Williams')
        st.write('10')





## team stats
st.header('🏆 Team Wins stat')
df = pd.read_csv('basketball.csv')
# st.line_chart(data=df, x='Season', y=['W', 'L'], x_label='Seasons (1980 - 1981 to 2024 - 2025)', y_label='Number of wins per season',
#               color=["#0099ff", "#ff0000ff"])
# st.line_chart(data=df, x='Season', y='W', x_label='Seasons (1980 - 1981 to 2024 - 2025)', y_label='Number of wins per season')

tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])

tab1.subheader("Team Wins per Season chart")
tab1.line_chart(data=df, x='Season', y='W', x_label='Seasons (1980 - 1981 to 2024 - 2025)', y_label='Number of wins per season')

tab2.subheader("Team Wins per Season data")
tab2.write(df)



## Predict 
# Load the pickled model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: model.pkl not found. Make sure the file is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# ข้อมูล feature และคำอธิบาย
feature_info = {
    "Finish": "อันดับจบฤดูกาล",
    "Age": "อายุเฉลี่ยของผู้เล่น (ปี)",
    "Ht. (ft)": "ส่วนสูงเฉลี่ย (ฟุต)",
    "Wt. (lbs)": "น้ำหนักเฉลี่ย (ปอนด์)",
    "G": "จำนวนเกมที่เล่น",
    "MP": "นาทีที่เล่นรวมทั้งหมด",
    "FG": "จำนวนครั้งที่ยิง",
    "FGA": "จำนวนครั้งที่ยิงลงห่วง",
    "FG%": "ความแม่นยำในการยิงลงห่วง",
    "3P": "จำนวนครั้งที่ยิงสามแต้ม",
    "3PA": "จำนวนครั้งที่ยิงสามแต้มลง",
    "3P%": "ความแม่นยำในการยิงสามแต้ม",
    "2P": "จำนวนครั้งที่ยิงสองแต้ม",
    "2PA": "จำนวนครั้งที่ยิงสองแต้มลง",
    "2P%": "ความแม่นยำในการยิงสองแต้ม",
    "FT": "จำนวนลูกโทษที่ยิง",
    "FTA": "จำนวนครั้งที่ยิงลูกโทษลง",
    "FT%": "ความแม่นยำในการยิงลูกโทษ",
    "ORB": "รีบาวน์เชิงรุก",
    "DRB": "รีบาวน์เชิงรับ",
    "TRB": "รีบาวน์รวม",
    "AST": "แอสซิสต์",
    "STL": "การสตีล",
    "BLK": "การบล็อค",
    "TOV": "เทิร์นโอเวอร์",
    "PF": "ฟาวล์ส่วนบุคคล",
    "PTS": "คะแนนรวม"
}

# ค่า default
default_values = [3, 27.5, 6.6, 217, 82, 19730, 3443, 7194, 0.479, 
                  1020, 2801, 0.364, 2423, 4393, 0.552, 1458, 
                  1894, 0.77, 828, 2702, 3530, 2070, 636, 445, 
                  1151, 1458, 9364]

feature_names = list(feature_info.keys())

# scaler (เดโม: fit จากค่า default)
scaler = StandardScaler()
scaler.fit([default_values])

st.header("🏀 Basketball Win Prediction")
st.write('The result is calculated based on overall team stat, not on mental and emotional performance, which may result in inaccurate results.')

with st.form("prediction_form"):
    inputs = []
    cols = st.columns(3)

    for i, (name, val) in enumerate(zip(feature_names, default_values)):
        with cols[i % 3]:
            num = st.number_input(f"{name} ({feature_info[name]})", value=float(val), format="%.3f")
            inputs.append(num)

    submitted = st.form_submit_button("Predict")

    if submitted:
        # แปลงเป็น array และทำ standardize
        X = np.array([inputs])
        X_scaled = scaler.transform(X)

        # prediction
        prediction = model.predict(X_scaled)

        progress_text = "Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

        st.subheader("Prediction Result 🏆")
        st.write(f"The team is expected to win: **{prediction[0]:,.0f}** Games")
        








## References

st.header('References')
st.write('1. About team, https://en.wikipedia.org/wiki/Dallas_Mavericks')
st.write('2. Team players, https://www.mavs.com/team/roster/')
