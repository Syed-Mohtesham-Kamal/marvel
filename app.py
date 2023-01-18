import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomML
from streamlit_bokeh_events import streamlit_bokeh_events
import pickle

model = pickle.load(open('model.pkl','rb'))

def main():
    
    st_button = Button(label="Speak", width=100)
    html_temp = """
<link href="https://fonts.googleapis.com/css?family=Roboto" rel="stylesheet">
<link rel="stylesheet" href="main.css"/>
<title> MARVEL </title>
<style> body {
    background-color: #fff;
    font-family: "Roboto";
  }
  
  .title {
    font-size: 32px;  
    color: #9E9E9E;
    margin: 20px 0px;
    text-align: center;
  }
  
  .canvas {
    position: relative;
    display: block;
    margin: auto;
    width: 600px;
    height: 420px;
    border-radius: 5px;
    background: none;
  }
  
  .blue {
    background-color: #E53935;
    left: 135px;
    animation-name: listening, voice-active-blue, pre-rotation-blue, rotation;
    animation-timing-function: ease-in-out, ease-in-out, ease-in-out, linear;
    animation-duration: 1.8s, 1.8s, .8s, 3s;
    animation-delay: 0s, 5.4s, 7.3s, 7.6s;
    animation-iteration-count: 3, 1, 1, infinite;
    animation-fill-mode: none, none, forwards, none;
  }
  
  .red {
    background-color: #E53935;
    left: 230px;
    animation-name: listening, voice-active-red, pre-rotation-red, rotation;
    animation-timing-function: ease-in-out, ease-in-out, ease-in-out, linear;
    animation-duration: 1.8s, 1.8s, .7s, 3s;
    animation-delay: 0.3s, 5.5s, 7.4s, 7.6s;
    animation-iteration-count: 3, 1, 1, infinite;
    animation-fill-mode: none, none, forwards, none;
  }
  
  .yellow {
    background-color: #E53935;
    left: 325px;
    animation-name: listening, voice-active-yellow, pre-rotation-yellow, rotation;
    animation-timing-function: ease-in-out, ease-in-out, ease-in-out, linear;
    animation-duration: 1.8s, 1.8s, .8s, 3s;
    animation-delay: 0.6s, 5.6s, 7.4s, 7.6s;
    animation-iteration-count: 3, 1, 1, infinite;
    animation-fill-mode: none, none, forwards, none;
  }
  
  .green {
    background-color: #E53935;
    left: 420px;
    animation-name: listening, voice-active-green, pre-rotation-green, rotation;
    animation-timing-function: ease-in-out, ease-in-out, ease-in-out, linear;
    animation-duration: 1.8s, 1.8s, .8s, 3s;
    animation-delay: 0.9s, 5.7s, 7.2s, 7.6s;
    animation-iteration-count: 3, 1, 1, infinite;
    animation-fill-mode: none, none, forwards, none;
  }
  
  .dot {
    position: absolute;
    height: 45px;
    width: 45px;
    border-radius: 50%;
    top: 200px;
  }
  
  @keyframes listening {
    0% {
      transform: translateY(0px);
    }
    
    50% {
      transform: translateY(25px);
    }
    
    100% {
      transform: translateY(0px);
    }
  }
  
  @keyframes voice-active-blue {
    0% {
      top: 200px;
      border-radius: 30px;
    }
    10% {
      top: 75px;
      padding: 125px 0px;
    }
    15% {
      top: 100px;
      padding: 100px 0px;
    }
    20% {
      top: 110px;
      padding: 90px 0px;
    }
    25% {
      top: 90px;
      padding: 110px 0px;
    }
    30% {
      top: 130px;
      padding: 70px 0px;
    }
    35% {
      top: 110px;
      padding: 90px 0px;
    }
    40% {
      top: 150px;
      padding: 50px 0px;
    }
    45% {
      top: 130px;
      padding: 70px 0px;
    }
    50% {
      top: 100px;
      padding: 100px 0px;
    }
    55% {
      top: 90px;
      padding: 110px 0px;
    }
    60% {
      top: 130px;
      padding: 70px 0px;
    }
    65% {
      top: 110px;
      padding: 90px 0px;
    }
    70% {
      top: 150px;
      padding: 50px 0px;
    }
    75% {
      top: 110px;
      padding: 90px 0px;
    }
    80% {
      top: 90px;
      padding: 110px 0px;
    }
    85% {
      top: 150px;
      padding: 50px 0px;
    }
    90%{
      top: 200px;  
      padding: 0;
    }
    100% {
      top: 200px;  
      padding: 0;
      border-radius: 30px;
    }
  }
  
  @keyframes voice-active-red {
    0% {
      top: 200px;
      border-radius: 30px;
    }
    10% {
      top: 145px;
      padding: 55px 0px;
    }
    15% {
      top: 165px;
      padding: 35px 0px;
    }
    20% {
      top: 145px;
      padding: 55px 0px;
    }
    25% {
      top: 155px;
      padding: 45px 0px;
    }
    30% {
      top: 175px;
      padding: 25px 0px;
    }
    35% {
      top: 155px;
      padding: 45px 0px;
    }
    40% {
      top: 165px;
      padding: 35px 0px;
    }
    45% {
      top: 175px;
      padding: 25px 0px;
    }
    50% {
      top: 165px;
      padding: 35px 0px;
    }
    55% {
      top: 180px;
      padding: 20px 0px;
    }
    60% {
      top: 185px;
      padding: 15px 0px;
    }
    65% {
      top: 155px;
      padding: 45px 0px;
    }
    70% {
      top: 175px;
      padding: 25px 0px;
    }
    75% {
      top: 155px;
      padding: 45px 0px;
    }
    80% {
      top: 180px;
      padding: 20px 0px;
    }
    85% {
      top: 190px;
      padding: 10px 0px;
    }
    90%{
      top: 200px;  
      padding: 0;
    }
    100% {
      top: 200px;  
      padding: 0;
      border-radius: 30px;
    }
  }
  
  
  @keyframes voice-active-yellow {
    0% {
      top: 200px;
      border-radius: 30px;
    }
    10% {
      top: 65px;
      padding: 135px 0px;
    }
    15% {
      top: 90px;
      padding: 110px 0px;
    }
    20% {
      top: 100px;
      padding: 100px 0px;
    }
    25% {
      top: 80px;
      padding: 120px 0px;
    }
    30% {
      top: 120px;
      padding: 80px 0px;
    }
    35% {
      top: 100px;
      padding: 100px 0px;
    }
    40% {
      top: 140px;
      padding: 60px 0px;
    }
    45% {
      top: 120px;
      padding: 80px 0px;
    }
    50% {
      top: 90px;
      padding: 110px 0px;
    }
    55% {
      top: 80px;
      padding: 120px 0px;
    }
    60% {
      top: 120px;
      padding: 80px 0px;
    }
    65% {
      top: 100px;
      padding: 100px 0px;
    }
    70% {
      top: 140px;
      padding: 60px 0px;
    }
    75% {
      top: 100px;
      padding: 100px 0px;
    }
    80% {
      top: 80px;
      padding: 120px 0px;
    }
    85% {
      top: 140px;
      padding: 60px 0px;
    }
    90%{
      top: 200px;  
      padding: 0;
    }
    100% {
      top: 200px;  
      padding: 0;
      border-radius: 30px;
    }
  }
  
  @keyframes voice-active-green {
    0% {
      top: 200px;
      border-radius: 30px;
    }
    10% {
      top: 135px;
      padding: 65px 0px;
    }
    15% {
      top: 155px;
      padding: 45px 0px;
    }
    20% {
      top: 135px;
      padding: 65px 0px;
    }
    25% {
      top: 145px;
      padding: 65px 0px;
    }
    30% {
      top: 165px;
      padding: 35px 0px;
    }
    35% {
      top: 145px;
      padding: 55px 0px;
    }
    40% {
      top: 155px;
      padding: 45px 0px;
    }
    45% {
      top: 165px;
      padding: 35px 0px;
    }
    50% {
      top: 155px;
      padding: 45px 0px;
    }
    55% {
      top: 170px;
      padding: 30px 0px;
    }
    60% {
      top: 175px;
      padding: 25px 0px;
    }
    65% {
      top: 145px;
      padding: 55px 0px;
    }
    70% {
      top: 165px;
      padding: 35px 0px;
    }
    75% {
      top: 145px;
      padding: 55px 0px;
    }
    80% {
      top: 170px;
      padding: 30px 0px;
    }
    85% {
      top: 190px;
      padding: 10px 0px;
    }
    90%{
      top: 200px;  
      padding: 0;
    }
    100% {
      top: 200px;  
      padding: 0;
      border-radius: 30px;
    }
  }
  
  
  @keyframes rotation {
    0% {
      transform: rotate(0deg);
    }
    50% {
    }
    100% {
      transform: rotate(360deg);
    }
  }
  
  @keyframes pre-rotation-blue {
    90% {
      top: 153px;
    }
    
    100%{
      transform-origin: 69px 69.5px;
      top: 153px;
      left: 227.5px;
    }
  }
  
  @keyframes pre-rotation-red {
    50%{
      top: 113px;
    }
    
    100%{
      transform-origin: -25px 69.5px;
      top: 153px;
      left: 322.5px;
    }
  }
  
  @keyframes pre-rotation-yellow {
    to{
      transform-origin: -25.5px -25.5px;
      top: 247.5px;
      left: 322.5px;
    }
  }
  
  @keyframes pre-rotation-green {
    50%{
      top: 287.5px;
    }
    100%{
      transform-origin: 69.5px -25.5px;
      top: 247.5px;
      left: 227.5px;
    }
  }
  
  @keyframes rotation-acceleration {
    from {}
    to{
      transform: rotate(180deg);
    }
  }
  
  </style>
<div class="title">M.A.R.V.E.L</div>

<div class="canvas">
  <div class="dot blue"></div>
  <div class="dot red"></div>
  <div class="dot yellow"></div>
  <div class="dot green"></div>
</div>
<div class='console-container'>
  <span id='text'></span>
<button type= "submit"style = "text-align:center;   background-color: #E53935; border: #E53935; border-radius: 20px;" > Speak </button>
<div><a>      </a></div>
<div>
  <a class="title" > Hello I am Marvel </a>
</div>
</div>  """

    st.markdown(html_temp, unsafe_allow_html=True)
    
    st_button.js_on_event("button_click", CustomML(code="model.pkl"))
    
    result = streamlit_bokeh_events(
    st_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)
    
    if result:
    if "GET_TEXT" in result:
        st.write(result.get("GET_TEXT"))
        

if __name__=='__main__':
    main()