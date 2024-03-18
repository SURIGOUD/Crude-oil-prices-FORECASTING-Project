
import streamlit as st
import pickle
import prophet
from prophet import Prophet
from prophet.plot import plot_plotly


pickle_in = open("prophet.pkl","rb")
model = pickle.load(pickle_in)

@st. cache_data()
def prediction(n_years):
  future = model.make_future_dataframe(periods = n_years*365)
  forecast = model.predict(future)
  data = forecast.iloc[n_years*(-365):]
  return data

def main():
  html_temp="""<div style = "background-color:indigo;padding:11px"> <h1 style = "color:white;text-align:center;">Crude Oil Prices Forecast</h1>
  </div>"""
  st.markdown(html_temp,unsafe_allow_html=True)
  n_years = st.slider("Select the number of years of forecast: ",1,5)
  if st.button("Forecast"):
    result = prediction(n_years)
    st.subheader("Forecast Data:")
    st.write(result[["ds","yhat"]])
    st.subheader("Forecast Plot:")
    fig = plot_plotly(model,result)
    st.plotly_chart(fig)

if __name__ =="__main__":
  main()
