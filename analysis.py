import streamlit as st
import pandas as pd
import numpy as np

attendence = pd.read_csv("Record.csv")
attendence=attendence.dropna()
finalAttendence=pd.read_csv("sample record.csv")
print(finalAttendence)
finalAttendence=finalAttendence.dropna()
if st.button("View Attendance",key="random"):
    # st.table(attendence)
    st.table(attendence)


# df = pd.read_csv("sample_record.csv")
# print(df)
# clock_in = df["Clock In Time"].str.split(':', expand=True).astype(int)
# df["Clock In Time"] = pd.to_timedelta(clock_in[0], unit='h')+ pd.to_timedelta(clock_in[1], unit='m')+ pd.to_timedelta(clock_in[2], unit='S')
# df["Date"] = pd.to_datetime(df["Date"],format="%d-%m-%Y")
# temp = df.set_index(["Name","Date"])["Clock In Time"]
# test = temp.groupby(level=[0,1]).agg("min")
# test2 = test.reset_index()#.astype({"Clock In Time":})
# test2["Clock In Time"]=test2["Clock In Time"].astype("timedelta64[h]")
# plot_data = test2.groupby("Name")
# st.line_chart(plot_data.get_group("Shlok")[["Date","Clock In Time"]].set_index("Date"))
