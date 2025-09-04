import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Loan Calculator Functions
# -------------------------------
def calculate_monthly_payment(principal, annual_rate, years):
    monthly_rate = annual_rate / 100 / 12
    months = years * 12
    if monthly_rate == 0:
        return principal / months
    return principal * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)

def generate_amortization_schedule(principal, annual_rate, years):
    monthly_payment = calculate_monthly_payment(principal, annual_rate, years)
    months = years * 12
    balance = principal
    schedule = []

    for m in range(1, months + 1):
        interest = balance * (annual_rate / 100 / 12)
        principal_payment = monthly_payment - interest
        balance -= principal_payment
        schedule.append([m, monthly_payment, principal_payment, interest, max(balance, 0)])

    df = pd.DataFrame(schedule, columns=["Month", "Payment", "Principal", "Interest", "Balance"])
    return df

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Loan Calculator", page_icon="💰", layout="wide")

st.title("💰 Interactive Loan Calculator")

# User Inputs
name = st.text_input("Enter your Name")
age = st.number_input("Enter your Age", min_value=18, max_value=100, value=25)
deposit = st.number_input("Deposit / Loan Amount", min_value=1000, step=1000)
interest = st.slider("Annual Interest Rate (%)", min_value=0.0, max_value=30.0, value=7.5, step=0.1)
duration = st.slider("Loan Duration (Years)", min_value=1, max_value=30, value=10)
show_schedule = st.checkbox("Show Full Amortization Schedule")

# Extra Options
toggle_extra = st.toggle("Show Extra Options")
if toggle_extra:
    st.write("Here you could add advanced features like insurance, tax, or prepayments.")

# -------------------------------
# Calculate Loan
# -------------------------------
if deposit > 0:
    monthly_payment = calculate_monthly_payment(deposit, interest, duration)
    total_payment = monthly_payment * duration * 12
    total_interest = total_payment - deposit

    st.subheader("📊 Loan Summary")
    st.write(f"**Name:** {name if name else 'N/A'} | **Age:** {age}")
    st.write(f"**Loan Amount:** ₹{deposit:,.2f}")
    st.write(f"**Monthly Payment:** ₹{monthly_payment:,.2f}")
    st.write(f"**Total Payment:** ₹{total_payment:,.2f}")
    st.write(f"**Total Interest Paid:** ₹{total_interest:,.2f}")

    # Amortization Data
    df = generate_amortization_schedule(deposit, interest, duration)

    # Line Chart (Principal vs Interest vs Balance)
    st.subheader("📈 Loan Balance Over Time")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["Month"], df["Balance"], label="Remaining Balance", color="blue")
    ax.plot(df["Month"], df["Principal"].cumsum(), label="Total Principal Paid", color="green")
    ax.plot(df["Month"], df["Interest"].cumsum(), label="Total Interest Paid", color="red")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (₹)")
    ax.legend()
    st.pyplot(fig)

    # Pie Chart (Principal vs Interest)
    st.subheader("🥧 Payment Distribution")
    fig2, ax2 = plt.subplots()
    ax2.pie([deposit, total_interest], labels=["Principal", "Interest"], autopct="%1.1f%%", startangle=90)
    ax2.axis("equal")
    st.pyplot(fig2)

    # Show full schedule if selected
    if show_schedule:
        st.subheader("📑 Amortization Schedule")
        st.dataframe(df, use_container_width=True)

