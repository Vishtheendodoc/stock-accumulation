import streamlit as st
import pandas as pd
import sqlite3
import requests
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import json
from io import StringIO
import plotly.express as px
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO
import numpy as np

st.set_page_config(layout="wide", page_title="NSE F&O Participant Data Dashboard")
# Telegram Bot Credentials

TELEGRAM_BOT_TOKEN = "7512763823:AAHwJN9YplSKy30gnIFZT5zIzBCZVYDsWLw"
TELEGRAM_CHAT_ID = "-4690137264"

# Database setup
db_path = "bhavcopy_data.db"

PREV_RATIOS_FILE = "previous_ratios.json"  # Stores previous ratios for comparison

def escape_markdown(text):
    """Escape special characters for Telegram MarkdownV2."""
    escape_chars = "_*[]()~`>#+-=|{}.!'"
    return "".join("\\" + char if char in escape_chars else char for char in text)

def send_telegram_alert(message):
    """Send message to Telegram with MarkdownV2 formatting."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": escape_markdown(message),
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True
    }

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"⚠️ Failed to send Telegram alert: {response.text}")

def load_previous_ratios():
    """Load stored previous day's ratios."""
    if os.path.exists(PREV_RATIOS_FILE):
        with open(PREV_RATIOS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_previous_ratios(ratios):
    import json  # <-- Add import inside function to be extra sure
    print("Saving previous ratios:", ratios)  # Debugging
    with open("previous_ratios.json", "w") as f:
        json.dump(ratios, f)



# Create table if not exists
def setup_database():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS bhavcopy (
                            date TEXT,
                            symbol TEXT,
                            series TEXT,
                            deliv_per REAL,
                            deliv_qty INTEGER,
                            ttl_trd_qnty INTEGER,
                            no_of_trades INTEGER,  -- ✅ Added column here
                            close_price REAL,
                            PRIMARY KEY (date, symbol)
                        )''')
        conn.commit()

setup_database()  # Ensure DB is set up at the start

# Function to check if a date is a trading day
def is_trading_day(date):
    return date.weekday() < 5  # Monday to Friday are trading days

# Function to download Bhavcopy
def download_bhavcopy(date):
    """Download NSE Bhavcopy CSV. If today's file is missing, fetch the latest available file."""
    attempts = 3  # Maximum attempts to find the latest available Bhavcopy
    
    while attempts > 0:
        date_str = date.strftime("%d%m%Y")  # Correct format
        url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
        file_path = f"bhavcopy_{date_str}.csv"

        if os.path.exists(file_path):
            return file_path  # Use cached file

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                st.success(f"✅ Successfully downloaded Bhavcopy for {date.strftime('%d-%m-%Y')}")
                return file_path
            else:
                print(f"⚠️ Bhavcopy not available for {date.strftime('%d-%m-%Y')}. Trying previous trading day...")
                date -= timedelta(days=1)  # Move to the previous day
                while not is_trading_day(date):  # Skip weekends
                    date -= timedelta(days=1)
                attempts -= 1  # Reduce attempts
        except requests.RequestException as e:
            st.error(f"❌ Error downloading Bhavcopy: {e}")
            return None

    st.warning("❌ No recent Bhavcopy available.")
    return None

def check_deliv_trade_ratio():
    """Check delivery/trade ratio for today compared to historical average."""
    today_str = datetime.today().strftime("%Y-%m-%d")
    ALERTS_TRACKING_FILE = "daily_delivery_ratio_alerts.json"
    
    # Prepare to track alerts
    existing_alerts = {}
    if os.path.exists(ALERTS_TRACKING_FILE):
        with open(ALERTS_TRACKING_FILE, 'r') as f:
            try:
                existing_alerts = json.load(f)
            except json.JSONDecodeError:
                existing_alerts = {}

    with sqlite3.connect(db_path) as conn:
        query = f'''
            SELECT date, symbol, deliv_qty, no_of_trades,
                   CAST(deliv_qty AS FLOAT) / no_of_trades AS ratio
            FROM bhavcopy
            WHERE date >= date('{today_str}', '-90 days')
            ORDER BY symbol, date
        '''
        df = pd.read_sql(query, conn)

    if df.empty:
        print("❌ No data found in database.")
        return pd.DataFrame()

    # Convert dates and ensure numeric ratios
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')

    # Calculate historical average ratio for each stock
    avg_ratios = df[df['date'] < datetime.today().date()].groupby('symbol')['ratio'].mean().reset_index()
    avg_ratios.columns = ['symbol', 'avg_ratio']

    # Get today's data
    today_data = df[df['date'] == datetime.today().date()]

    # Merge today's data with historical averages
    merged_df = today_data.merge(avg_ratios, on='symbol', how='left')

    # Calculate ratio compared to average
    merged_df['ratio_vs_avg'] = merged_df['ratio'] / merged_df['avg_ratio']

    # Filter stocks with ratio > 1.5 times average
    alert_df = merged_df[merged_df['ratio_vs_avg'] > 1.5].copy()

    print("🔍 Filtered Stocks with Today's Ratio > 1.5x Historical Average:")
    print(alert_df[['date', 'symbol', 'ratio', 'avg_ratio', 'ratio_vs_avg', 'deliv_qty', 'no_of_trades']])

    if alert_df.empty:
        print("⚠️ No stocks found with today's ratio > 1.5x average.")
        return df

    # Prepare Telegram Alert Messages
    alert_messages = [f"🚀 *Stocks with High Delivery/Trades Ratio on {today_str}:*\n"]
    
    # Track new alerts to be sent
    new_alerts_to_send = []
    updated_alerts = existing_alerts.copy()
    
    for _, row in alert_df.iterrows():
        # Create a unique key for this alert
        alert_key = row['symbol']
        
        # Prepare alert details
        alert_details = {
            'date': today_str,
            'ratio': round(row['ratio'], 2),
            'avg_ratio': round(row['avg_ratio'], 2),
            'ratio_vs_avg': round(row['ratio_vs_avg'], 2),
            'deliv_qty': row['deliv_qty'],
            'no_of_trades': row['no_of_trades']
        }
        
        # Check if this alert is new or different from previous
        if (alert_key not in existing_alerts or 
            existing_alerts[alert_key] != alert_details):
            
            message = (f"🔹 *{escape_markdown(row['symbol'])}* (📅 {row['date']})\n"
                       f"   - 📈 Today's Ratio: `{alert_details['ratio']:.2f}`\n"
                       f"   - 📊 Historical Avg Ratio: `{alert_details['avg_ratio']:.2f}`\n"
                       f"   - 🚀 Ratio vs Avg: `{alert_details['ratio_vs_avg']:.2f}x`\n"
                       f"   - 📦 DELIV_QTY: `{alert_details['deliv_qty']}`\n"
                       f"   - 📊 NO_OF_TRADES: `{alert_details['no_of_trades']}`\n")
            alert_messages.append(message)
            
            # Update the alerts tracking
            updated_alerts[alert_key] = alert_details
            new_alerts_to_send.append(row['symbol'])

    # Send alerts if there are new ones
    if len(alert_messages) > 1:  # More than just the initial header
        # Handle Telegram message length limit (split into chunks)
        max_length = 4000  
        message_chunk = ""
        for line in alert_messages:
            if len(message_chunk) + len(line) > max_length:
                send_telegram_alert(message_chunk)
                message_chunk = ""  
            message_chunk += line + "\n"

        if message_chunk:
            send_telegram_alert(message_chunk)
        
        print(f"🔔 Sent new alerts for: {', '.join(new_alerts_to_send)}")
    else:
        print("⚠️ No new alerts to send.")

    # Save updated alerts tracking
    with open(ALERTS_TRACKING_FILE, 'w') as f:
        json.dump(updated_alerts, f)

    # Save today's ratios for future comparison
    save_previous_ratios({row["symbol"]: row["ratio"] for _, row in df.iterrows()})

    return df

# Function to process and store data
def process_bhavcopy(file_path, date, db_path):
    try:
        df = pd.read_csv(file_path)
        
        # ✅ Normalize column names
        df.columns = df.columns.str.strip().str.upper()  

        # ✅ Filter dataframe to keep only specified symbols
        symbols_list = [
            'GODREJCP', 'ADANIENSOL', 'FEDERALBNK', 'SYNGENE', 'LUPIN', 'BHARTIARTL',
            'PNB', 'OFSS', 'UNIONBANK', 'CANBK', 'DIVISLAB', 'MARUTI', 'IDFCFIRSTB',
            'INDIGO', 'JSL', 'IGL', 'UPL', 'PETRONET', 'BIOCON', 'LT', 'ULTRACEMCO',
            'TCS', 'COFORGE', 'NTPC', 'ALKEM', 'JSWSTEEL', 'GRANULES', 'GRASIM',
            'AMBUJACEM', 'ABB', 'JKCEMENT', 'M&MFIN', 'KEI', 'HDFCBANK', 'HEROMOTOCO',
            'TECHM', 'IRCTC', 'HINDALCO', 'NAUKRI', 'TORNTPOWER', 'HINDUNILVR',
            'HINDPETRO', 'PFC', 'INDIANB', 'IDEA', 'PAGEIND', 'ASTRAL', 'DLF',
            'POWERGRID', 'ADANIPORTS', 'INDHOTEL', 'COLPAL', 'NMDC', 'RECLTD',
            'APOLLOHOSP', 'INFY', 'IOC', 'APOLLOTYRE', 'ITC', 'BAJAJ-AUTO',
            'BAJAJFINSV', 'CROMPTON', 'OIL', 'MGL', 'JSWENERGY', 'GODREJPROP',
            'HINDCOPPER', 'NESTLEIND', 'SBICARD', 'JUBLFOOD', 'JIOFIN', 'PERSISTENT',
            'RBLBANK', 'LTTS', 'ICICIPRULI', 'VBL', 'CUMMINSIND', 'MANAPPURAM',
            'KOTAKBANK', 'LAURUSLABS', 'BSE', 'TRENT', 'DMART', 'DEEPAKNTR',
            'LICHSGFIN', 'OBEROIRLTY', 'IRFC', 'PRESTIGE', 'M&M', 'COALINDIA',
            'RAMCOCEM', 'HUDCO', 'CDSL', 'ASHOKLEY', 'AUBANK', 'MFSL', 'ABCAPITAL',
            'DIXON', 'ICICIGI', 'BOSCHLTD', 'ACC', 'IEX', 'BANDHANBNK', 'MRF', 'HAL',
            'ASIANPAINT', 'MUTHOOTFIN', 'PEL', 'PIIND', 'ONGC', 'ADANIENT',
            'APLAPOLLO', 'PIDILITIND', 'AUROPHARMA', 'RELIANCE', 'INDUSTOWER',
            'KALYANKJIL', 'SAIL', 'ABFRL', 'SBIN', 'VEDL', 'SHREECEM', 'MCX',
            'SIEMENS', 'BAJFINANCE', 'ANGELONE', 'SRF', 'SUNPHARMA', 'TATACHEM',
            'TATAELXSI', 'TATAPOWER', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TITAN',
            'TORNTPHARM', 'ADANIGREEN', 'VOLTAS', 'TATACOMM', 'WIPRO', 'BEL',
            'BERGEPAINT', 'MARICO', 'MOTHERSON', 'BHARATFORG', 'HDFCAMC', 'BHEL',
            'MPHASIS', 'BANKBARODA', 'HDFCLIFE', 'GAIL', 'CONCOR', 'ICICIBANK',
            'INDUSINDBK', 'BRITANNIA', 'AXISBANK', 'NATIONALUM', 'CHAMBLFERT',
            'JINDALSTEL', 'EXIDEIND', 'CHOLAFIN', 'CIPLA', 'BSOFT', 'AARTIIND',
            'HCLTECH', 'GLENMARK', 'DABUR', 'ZYDUSLIFE', 'TVSMOTOR', 'DRREDDY',
            'EICHERMOT', 'ESCORTS', 'POLYCAB', 'HAVELLS'
        ]

        # ✅ Filter only selected stocks
        df = df[df['SYMBOL'].isin(symbols_list)]

        # ✅ Trim whitespace from string values
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # ✅ Rename DATE1 → DATE if present
        if "DATE1" in df.columns:
            df.rename(columns={"DATE1": "DATE"}, inplace=True)

        # ✅ Convert DATE to YYYY-MM-DD format
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")

        # ✅ Check for missing columns
        required_cols = {"SYMBOL", "SERIES", "DELIV_PER", "DELIV_QTY", "TTL_TRD_QNTY", "CLOSE_PRICE", "NO_OF_TRADES", "DATE"}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            st.error(f"❌ Missing columns in Bhavcopy file: {missing_cols}")
            return None

        # ✅ Keep only EQ series stocks
        df = df[df["SERIES"] == "EQ"]

        # ✅ Convert DELIV_PER properly
        df["DELIV_PER"] = df["DELIV_PER"].astype(str).str.replace(",", "").str.replace("%", "").astype(float)
        df["DELIV_QTY"] = pd.to_numeric(df["DELIV_QTY"], errors="coerce")

        # ✅ Remove rows with NaN delivery percentage
        df = df.dropna(subset=["DELIV_PER"])

        # ✅ Filter stocks with high delivery percentage (>60%)
        df = df[df["DELIV_PER"] > 60]

        if df.empty:
            st.warning("⚠️ No stocks found with DELIV_PER > 60%. Try lowering the threshold.")
            return None

        # ✅ Select only required columns
        df = df[["DATE", "SYMBOL", "SERIES", "DELIV_PER", "DELIV_QTY", "TTL_TRD_QNTY", "CLOSE_PRICE", "NO_OF_TRADES"]]

        # ✅ Insert into SQLite database (avoid duplicates)
        with sqlite3.connect(db_path) as conn:
            existing_records = pd.read_sql("SELECT date, symbol FROM bhavcopy", conn)
            df = df[~df[["DATE", "SYMBOL"]].apply(tuple, axis=1).isin(existing_records.apply(tuple, axis=1))]
            if not df.empty:
                df.to_sql("bhavcopy", conn, if_exists="append", index=False)

        return df

    except Exception as e:
        st.error(f"❌ Error processing Bhavcopy file: {e}")
        return None

# Function to get accumulation stocks
def get_accumulation_stocks(days=5):
    try:
        with sqlite3.connect(db_path) as conn:
            query = f'''
                SELECT symbol, 
                       AVG(deliv_per) AS avg_deliv_per, 
                       AVG(deliv_qty) AS avg_deliv_qty, 
                       AVG(no_of_trades) AS avg_trades
                FROM bhavcopy
                WHERE date >= date('now', '-{days} days')
                GROUP BY symbol
                HAVING avg_deliv_per > 60 AND avg_trades > 1  -- ✅ Only include liquid stocks
                ORDER BY avg_deliv_qty DESC
            '''
            return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Error fetching accumulation stocks: {e}")
        return pd.DataFrame()

# Function to fetch F&O Participant OI data from NSE archives
def fetch_fo_participant_oi(date):
    """Fetch F&O participant OI data CSV for a given date."""
    url = f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{date}.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text  # Return CSV content
        else:
            st.error(f"Failed to fetch data for {date}: {response.status_code}")
            return None
    except requests.RequestException as e:
        st.error(f"Request error: {e}")
        return None

# Clean and process the CSV data
def clean_participant_oi_data(csv_content, date_str):
    """Clean and structure the participant OI data."""
    try:
        # Read CSV content
        df = pd.read_csv(StringIO(csv_content), skiprows=1)
        
        # Extract only the relevant data
        # Find rows with client types
        client_types = ['Client', 'DII', 'FII', 'Pro', 'TOTAL']
        
        # Find client type column
        client_type_col = df.columns[0]
        
        # Filter rows with valid client types
        df = df[df[client_type_col].isin(client_types)]
        
        # Rename columns properly
        column_names = [
            'Client_Type', 'Future_Index_Long', 'Future_Index_Short', 
            'Future_Stock_Long', 'Future_Stock_Short',
            'Option_Index_Call_Long', 'Option_Index_Put_Long', 
            'Option_Index_Call_Short', 'Option_Index_Put_Short',
            'Option_Stock_Call_Long', 'Option_Stock_Put_Long', 
            'Option_Stock_Call_Short', 'Option_Stock_Put_Short',
            'Total_Long_Contracts', 'Total_Short_Contracts'
        ]
        
        # Use only up to the number of columns we want to rename
        actual_cols = min(len(df.columns), len(column_names))
        rename_dict = {df.columns[i]: column_names[i] for i in range(actual_cols)}
        df = df.rename(columns=rename_dict)
        
        # Convert columns to numeric
        for col in df.columns:
            if col != 'Client_Type':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add date column in proper format
        df['Date'] = pd.to_datetime(date_str, format='%d%m%Y').strftime('%Y-%m-%d')
        
        return df
    except Exception as e:
        st.error(f"Error processing data for {date_str}: {str(e)}")
        return None

# Function to fetch data for the last N trading days
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_trading_days_oi(days=15):
    """Fetch F&O participant OI for the specified number of trading days."""
    oi_data = []
    date = datetime.today()
    count = 0
    attempts = 0
    max_attempts = days * 3  # Allow more attempts for weekends/holidays
    
    progress_bar = st.progress(0)
    
    while count < days and attempts < max_attempts:
        date_str = date.strftime("%d%m%Y")  # Format: DDMMYYYY
        
        # Update progress status
        status_text = st.empty()
        status_text.text(f"Fetching data for {date.strftime('%Y-%m-%d')}...")
        
        csv_content = fetch_fo_participant_oi(date_str)
        
        if csv_content:
            cleaned_df = clean_participant_oi_data(csv_content, date_str)
            if cleaned_df is not None and not cleaned_df.empty:
                oi_data.append(cleaned_df)
                count += 1
                progress_bar.progress(count / days)
                status_text.text(f"Successfully processed data for {date.strftime('%Y-%m-%d')} ({count}/{days})")
        
        # Move to the previous day
        date -= timedelta(days=1)
        attempts += 1
    
    progress_bar.empty()
    status_text.empty()
    
    if oi_data:
        # Combine all dataframes
        result_df = pd.concat(oi_data, ignore_index=True)
        return result_df
    else:
        st.error("No data was collected. Please check if the NSE website is accessible.")
        return None

# Streamlit UI
st.title("NSE Bhavcopy Analysis - High Delivery & Accumulation")

# Fetch last 30 days' data
days = 30
for i in range(days):
    date = datetime.today() - timedelta(days=i)

    file_path = download_bhavcopy(date)
    
    if file_path:
        df = process_bhavcopy(file_path, date, db_path)  # ✅ Corrected
        if df is not None:
            st.write(f"✅ Processed data for {date.strftime('%Y-%m-%d')} successfully.")
        else:
            st.warning(f"❌ Processing failed for {date.strftime('%Y-%m-%d')}.")

# 🔹 Store the ratio data for plotting
ratio_df = check_deliv_trade_ratio()  # ✅ Ensures data is available later

# Verify Database Insertion
with sqlite3.connect(db_path) as conn:
    verify_df = pd.read_sql("SELECT * FROM bhavcopy LIMIT 5", conn)
    st.write("🔹 Sample data from database:", verify_df)

# Display results
st.subheader("Stocks with High Delivery Percentage & Accumulation")
accumulation_df = get_accumulation_stocks(days)

if not accumulation_df.empty:
    st.dataframe(accumulation_df)
else:
    st.warning("No accumulation stocks found.")

# Plot accumulation trends
st.subheader("Accumulation Trend of Selected Stock")

# Fetch unique stocks from database
with sqlite3.connect(db_path) as conn:
    stock_list = pd.read_sql("SELECT DISTINCT symbol FROM bhavcopy", conn)["symbol"].tolist()

# User selects a stock from dropdown
selected_stock = st.selectbox("📌 Select a stock to view accumulation trend:", stock_list)

if selected_stock:
    with sqlite3.connect(db_path) as conn:
        trend_df = pd.read_sql(f"SELECT date, deliv_per, deliv_qty FROM bhavcopy WHERE symbol = '{selected_stock}' ORDER BY date", conn)

    if not trend_df.empty:
        # Convert 'date' column to proper datetime format
        trend_df["date"] = pd.to_datetime(trend_df["date"], format="%Y-%m-%d")
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        # Plot Delivery %
        ax1.plot(trend_df["date"], trend_df["deliv_per"], marker='o', linestyle='-', color='blue', label='Delivery %')
        
        # Plot Delivery Quantity
        ax2.plot(trend_df["date"], trend_df["deliv_qty"], marker='s', linestyle='--', color='red', label='Delivery Volume')
        
        # Set labels and title
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Delivery %", color='blue')
        ax2.set_ylabel("Delivery Volume", color='red')
        ax1.set_title(f"Accumulation Trend for {selected_stock}")
        
        # Create a custom date formatter that uses short month names with just 1 character
        def custom_date_fmt(x, pos=None):
            date = mdates.num2date(x)
            if date.day == 1:   # Show full format for 1st of month
                return date.strftime('%d-%b')
            else:
                return date.strftime('%d')  # Only show day number for other dates
        
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_fmt))
        
        # Rotate the labels
        plt.xticks(rotation=90, fontsize=8)
        
        # Add extra space at the bottom
        plt.subplots_adjust(bottom=0.2)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.grid()
        
        st.pyplot(fig)
    else:
        st.warning(f"No trend data available for {selected_stock}.")

# Replace the existing plotting section with this updated code
st.subheader("📊 Delivery to Trades Ratio Over Time")

# Fetch unique stock symbols from the database
with sqlite3.connect(db_path) as conn:
    stock_list = pd.read_sql("SELECT DISTINCT symbol FROM bhavcopy", conn)["symbol"].tolist()

# Select a stock from the dropdown
selected_stock = st.selectbox("📌 Select a stock to view ratio trend:", stock_list)

# Filter data for the selected stock
with sqlite3.connect(db_path) as conn:
    stock_data = pd.read_sql(f"""
        SELECT 
            date, 
            symbol, 
            deliv_qty, 
            no_of_trades,
            CAST(deliv_qty AS FLOAT) / NULLIF(no_of_trades, 0) AS ratio
        FROM bhavcopy 
        WHERE symbol = '{selected_stock}' 
        ORDER BY date
    """, conn)

if not stock_data.empty:
    # Convert date and handle potential errors
    stock_data["date"] = pd.to_datetime(stock_data["date"], errors="coerce")
    
    # Sort by date
    stock_data = stock_data.sort_values("date")
    
    # Calculate ratio change 
    stock_data["ratio_change"] = stock_data["ratio"].pct_change()

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ratio trend
    ax.plot(stock_data["date"], stock_data["ratio"], marker='o', linestyle='-', color='blue', label='DELIV_QTY / NO_OF_TRADES')
    
    # Highlight points with significant ratio change (e.g., more than 100% increase)
    doubled_data = stock_data[stock_data["ratio_change"] >= 1]
    if not doubled_data.empty:
        ax.scatter(doubled_data["date"], doubled_data["ratio"], color='red', label='Significant Ratio Change', zorder=3)

    # Labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Ratio (DELIV_QTY / NO_OF_TRADES)")
    ax.set_title(f"Trend of {selected_stock} Delivery to Trades Ratio")
    ax.legend()
    ax.grid(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    st.pyplot(fig)
else:
    st.warning(f"No ratio data available for {selected_stock}.")


# stock wise all table details
st.subheader("📊 Stock Data & Delivery Ratio Analysis")

# Fetch unique stock symbols from the database
with sqlite3.connect(db_path) as conn:
    stock_list = pd.read_sql("SELECT DISTINCT symbol FROM bhavcopy", conn)["symbol"].tolist()

# Select a stock from the dropdown
selected_stock = st.selectbox("📌 Select a stock to view details:", stock_list)

if selected_stock:
    with sqlite3.connect(db_path) as conn:
        stock_data = pd.read_sql(f"""
            SELECT 
                date, 
                symbol, 
                deliv_per, 
                deliv_qty, 
                no_of_trades, 
                close_price,
                CAST(deliv_qty AS FLOAT) / NULLIF(no_of_trades, 0) AS delivery_trade_ratio
            FROM bhavcopy 
            WHERE symbol = '{selected_stock}' 
            ORDER BY date DESC
        """, conn)

    if not stock_data.empty:
        # Convert 'date' to datetime format
        stock_data["date"] = pd.to_datetime(stock_data["date"], errors="coerce")

        # Round numeric columns for better readability
        numeric_cols = ['deliv_per', 'deliv_qty', 'no_of_trades', 'close_price', 'delivery_trade_ratio']
        stock_data[numeric_cols] = stock_data[numeric_cols].round(2)

        # Display as a table
        st.dataframe(stock_data)

        # Basic statistics
        st.subheader("📊 Stock Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Delivery %", f"{stock_data['deliv_per'].mean():.2f}%")
            st.metric("Average Delivery Qty", f"{stock_data['deliv_qty'].mean():.0f}")
        
        with col2:
            st.metric("Average Trades", f"{stock_data['no_of_trades'].mean():.0f}")
            st.metric("Avg Delivery/Trade Ratio", f"{stock_data['delivery_trade_ratio'].mean():.2f}")

    else:
        st.warning(f"No data available for {selected_stock}.")
        
# Real-time Alerts Section
st.subheader("🚨 Real-time Delivery/Trade Ratio Alerts")

# Function to get high ratio stocks
def get_high_ratio_stocks():
    try:
        with sqlite3.connect(db_path) as conn:
            today_str = datetime.today().strftime("%Y-%m-%d")
            query = f'''
                WITH RatioCalculation AS (
                    SELECT 
                        symbol, 
                        date,
                        deliv_qty, 
                        no_of_trades,
                        CAST(deliv_qty AS FLOAT) / no_of_trades AS ratio,
                        (
                            SELECT AVG(CAST(deliv_qty AS FLOAT) / no_of_trades) 
                            FROM bhavcopy 
                            WHERE symbol = b.symbol AND date < '{today_str}'
                        ) AS avg_ratio
                    FROM bhavcopy b
                    WHERE date = '{today_str}'
                )
                SELECT 
                    symbol, 
                    date, 
                    ratio, 
                    avg_ratio, 
                    (ratio / avg_ratio) AS ratio_vs_avg,
                    deliv_qty,
                    no_of_trades
                FROM RatioCalculation
                WHERE (ratio / avg_ratio) > 1.5
                ORDER BY (ratio / avg_ratio) DESC
            '''
            return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Error fetching high ratio stocks: {e}")
        return pd.DataFrame()

# Display real-time alerts
high_ratio_stocks = get_high_ratio_stocks()

if not high_ratio_stocks.empty:
    # Create color-coded alerts
    for _, stock in high_ratio_stocks.iterrows():
        alert_color = "red" if stock['ratio_vs_avg'] > 2 else "orange"
        
        # Create a container for each alert
        alert_container = st.container()
        with alert_container:
            st.markdown(f"""
            <div style="background-color:{alert_color}; color:white; padding:10px; border-radius:10px;">
            🚨 **{stock['symbol']}** Alert
            - 📈 Today's Ratio: `{stock['ratio']:.2f}`
            - 📊 Historical Avg Ratio: `{stock['avg_ratio']:.2f}`
            - 🚀 Ratio vs Avg: `{stock['ratio_vs_avg']:.2f}x`
            - 📦 Delivery Qty: `{stock['deliv_qty']}`
            - 📊 No. of Trades: `{stock['no_of_trades']}`
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("📊 No high delivery/trade ratio stocks found today.")

# Add this section to your Streamlit UI

# Manual High Ratio Alert Section
st.sidebar.header("🚨 Manual High Ratio Alerts")

# Function to get high ratio stocks
def get_manual_high_ratio_stocks():
    try:
        with sqlite3.connect(db_path) as conn:
            today_str = datetime.today().strftime("%Y-%m-%d")
            query = f'''
                WITH RatioCalculation AS (
                    SELECT 
                        symbol, 
                        date,
                        deliv_qty, 
                        no_of_trades,
                        close_price,
                        CAST(deliv_qty AS FLOAT) / no_of_trades AS ratio,
                        (
                            SELECT AVG(CAST(deliv_qty AS FLOAT) / no_of_trades) 
                            FROM bhavcopy 
                            WHERE symbol = b.symbol AND date < '{today_str}'
                        ) AS avg_ratio
                    FROM bhavcopy b
                    WHERE date = '{today_str}'
                )
                SELECT 
                    symbol, 
                    date, 
                    ratio, 
                    avg_ratio, 
                    (ratio / avg_ratio) AS ratio_vs_avg,
                    deliv_qty,
                    no_of_trades,
                    close_price
                FROM RatioCalculation
                WHERE (ratio / avg_ratio) > 1.5
                ORDER BY (ratio / avg_ratio) DESC
            '''
            return pd.read_sql(query, conn)
    except Exception as e:
        st.sidebar.error(f"Error fetching high ratio stocks: {e}")
        return pd.DataFrame()

# Fetch high ratio stocks
high_ratio_stocks = get_manual_high_ratio_stocks()

# Display high ratio stocks and allow manual alert
if not high_ratio_stocks.empty:
    st.sidebar.subheader("📊 High Ratio Stocks")
    
    # Multiselect for stocks
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks for Alert", 
        high_ratio_stocks['symbol'].tolist()
    )
    
    # Custom message prefix
    message_prefix = st.sidebar.text_input(
        "Alert Message Prefix", 
        value="🚀 High Delivery/Trade Ratio Alert:"
    )
    
    # Send selected stocks alert
    if st.sidebar.button("🔔 Send Selected Stock Alerts"):
        for symbol in selected_stocks:
            # Find the stock details
            stock_details = high_ratio_stocks[high_ratio_stocks['symbol'] == symbol].iloc[0]
            
            # Construct detailed alert message
            alert_message = (
                f"{message_prefix}\n"
                f"🔹 *{escape_markdown(symbol)}*\n"
                f"   - 📈 Today's Ratio: `{stock_details['ratio']:.2f}`\n"
                f"   - 📊 Historical Avg Ratio: `{stock_details['avg_ratio']:.2f}`\n"
                f"   - 🚀 Ratio vs Avg: `{stock_details['ratio_vs_avg']:.2f}x`\n"
                f"   - 📦 DELIV_QTY: `{stock_details['deliv_qty']}`\n"
                f"   - 📊 NO_OF_TRADES: `{stock_details['no_of_trades']}`\n"
                f"   - 💹 CLOSE_PRICE: `₹{stock_details['close_price']:.2f}`\n"
            )
            
            # Send Telegram alert
            try:
                send_telegram_alert(alert_message)
                st.sidebar.success(f"✅ Alert sent for {symbol}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to send alert for {symbol}: {e}")
else:
    st.sidebar.info("📊 No high delivery/trade ratio stocks found today.")

# Display in Streamlit
    st.title("NSE F&O Participant Open Interest Dashboard")
    
    st.sidebar.header("Dashboard Settings")
    days_to_fetch = st.sidebar.slider("Number of trading days to fetch", 5, 30, 15)
    
    # Fetch data
    with st.spinner("Fetching F&O participant data..."):
        oi_df = get_trading_days_oi(days=days_to_fetch)
    
    if oi_df is not None:
        # Add dropdown to select specific date
        available_dates = sorted(oi_df['Date'].unique(), reverse=True)
        selected_date = st.selectbox("Select Date for F&O Participant Data", options=available_dates)
        
        # Get previous date for comparison
        available_dates_sorted = sorted(oi_df['Date'].unique())
        selected_date_index = available_dates_sorted.index(selected_date)
        previous_date = available_dates_sorted[selected_date_index - 1] if selected_date_index > 0 else None
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Daily Change", "Time Series Analysis", "Long-Short Ratio"])
        
        with tab1:
            st.subheader(f"F&O Participant Data - {selected_date}")
            # Get the selected date data
            selected_data = oi_df[oi_df['Date'] == selected_date]
            
            # Exclude TOTAL row for the charts
            chart_data = selected_data[selected_data['Client_Type'] != 'TOTAL']
            
            # Create two columns for the layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Long positions chart
                fig_long = px.bar(
                    chart_data,
                    x='Client_Type',
                    y=['Future_Index_Long', 'Future_Stock_Long', 'Option_Index_Call_Long', 
                       'Option_Index_Put_Long', 'Option_Stock_Call_Long', 'Option_Stock_Put_Long'],
                    title=f"Long Positions by Participant Type ({selected_date})",
                    labels={'value': 'Contracts', 'variable': 'Position Type'},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                st.plotly_chart(fig_long)
            
            with col2:
                # Short positions chart
                fig_short = px.bar(
                    chart_data,
                    x='Client_Type',
                    y=['Future_Index_Short', 'Future_Stock_Short', 'Option_Index_Call_Short', 
                       'Option_Index_Put_Short', 'Option_Stock_Call_Short', 'Option_Stock_Put_Short'],
                    title=f"Short Positions by Participant Type ({selected_date})",
                    labels={'value': 'Contracts', 'variable': 'Position Type'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_short)
            
            # Display total long-short balance
            st.subheader("Total Position Summary")
            total_summary = selected_data[selected_data['Client_Type'] == 'TOTAL']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Long Contracts", f"{int(total_summary['Total_Long_Contracts'].values[0]):,}")
            with col2:
                st.metric("Total Short Contracts", f"{int(total_summary['Total_Short_Contracts'].values[0]):,}")
            with col3:
                long_short_ratio = total_summary['Total_Long_Contracts'].values[0] / total_summary['Total_Short_Contracts'].values[0]
                st.metric("Long/Short Ratio", f"{long_short_ratio:.2f}")
            
            # Raw data table with formatting
            st.subheader("Raw Data")
            st.dataframe(selected_data.style.format({col: '{:,.0f}' for col in selected_data.columns 
                                                 if col not in ['Client_Type', 'Date']}))
        
        with tab2:
            if previous_date:
                st.subheader(f"Daily Change Analysis ({selected_date} vs {previous_date})")
                
                # Get data for selected and previous day
                current_day_data = oi_df[oi_df['Date'] == selected_date].copy()
                previous_day_data = oi_df[oi_df['Date'] == previous_date].copy()
                
                # Prepare data for comparison
                current_day_data.set_index('Client_Type', inplace=True)
                previous_day_data.set_index('Client_Type', inplace=True)
                
                # Calculate changes
                numeric_columns = [col for col in current_day_data.columns if col not in ['Date']]
                change_data = current_day_data[numeric_columns] - previous_day_data[numeric_columns]
                change_data_pct = ((current_day_data[numeric_columns] - previous_day_data[numeric_columns]) / 
                                    previous_day_data[numeric_columns].abs()) * 100
                
                # Reset index for plotting
                change_data.reset_index(inplace=True)
                change_data_pct.reset_index(inplace=True)
                
                # Filter out TOTAL for charts
                change_chart_data = change_data[change_data['Client_Type'] != 'TOTAL']
                
                # Display absolute changes
                st.subheader("Absolute Changes in Positions")
                
                # Display changes by client type
                col1, col2 = st.columns(2)
                
                with col1:
                    # Net change in long positions
                    fig_change_long = px.bar(
                        change_chart_data,
                        x='Client_Type',
                        y='Total_Long_Contracts',
                        title="Change in Long Positions",
                        color='Total_Long_Contracts',
                        color_continuous_scale=px.colors.diverging.RdBu,
                        labels={'Total_Long_Contracts': 'Change in Contracts'}
                    )
                    fig_change_long.update_layout(coloraxis_colorbar=dict(title="Contracts"))
                    st.plotly_chart(fig_change_long)
                
                with col2:
                    # Net change in short positions
                    fig_change_short = px.bar(
                        change_chart_data,
                        x='Client_Type',
                        y='Total_Short_Contracts',
                        title="Change in Short Positions",
                        color='Total_Short_Contracts',
                        color_continuous_scale=px.colors.diverging.RdBu,
                        labels={'Total_Short_Contracts': 'Change in Contracts'}
                    )
                    fig_change_short.update_layout(coloraxis_colorbar=dict(title="Contracts"))
                    st.plotly_chart(fig_change_short)
                
                # Detailed position changes
                st.subheader("Detailed Position Changes")
                
                # Position type selector for detailed changes
                position_groups = {
                    "Futures - Index": ['Future_Index_Long', 'Future_Index_Short'],
                    "Futures - Stock": ['Future_Stock_Long', 'Future_Stock_Short'],
                    "Options - Index Calls": ['Option_Index_Call_Long', 'Option_Index_Call_Short'],
                    "Options - Index Puts": ['Option_Index_Put_Long', 'Option_Index_Put_Short'],
                    "Options - Stock Calls": ['Option_Stock_Call_Long', 'Option_Stock_Call_Short'],
                    "Options - Stock Puts": ['Option_Stock_Put_Long', 'Option_Stock_Put_Short']
                }
                
                selected_position_group = st.selectbox("Select Position Group", options=list(position_groups.keys()))
                selected_columns = position_groups[selected_position_group]
                
                # Create heatmap showing changes by client type
                change_pivot = change_data.melt(
                    id_vars=['Client_Type'],
                    value_vars=selected_columns,
                    var_name='Position_Type',
                    value_name='Change'
                )
                change_pivot = change_pivot[change_pivot['Client_Type'] != 'TOTAL']
                
                fig_heatmap = px.density_heatmap(
                    change_pivot,
                    x='Client_Type',
                    y='Position_Type',
                    z='Change',
                    title=f"Change Heatmap for {selected_position_group}",
                    color_continuous_scale=px.colors.diverging.RdBu,
                    labels={'Change': 'Contract Change'}
                )
                fig_heatmap.update_layout(coloraxis_colorbar=dict(title="Contract Change"))
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Summary metrics by participant type
            st.subheader("Key Change Metrics by Participant Type")

            # Get data for all participant types (excluding TOTAL)
            participant_types = [client for client in current_day_data.index if client != 'TOTAL']

            # Create a DataFrame to hold the changes
            change_summary = pd.DataFrame()

            for client in participant_types:
                # Calculate key metrics for each participant
                current_long = current_day_data.loc[client, 'Total_Long_Contracts']
                current_short = current_day_data.loc[client, 'Total_Short_Contracts']
                previous_long = previous_day_data.loc[client, 'Total_Long_Contracts']
                previous_short = previous_day_data.loc[client, 'Total_Short_Contracts']
                
                long_pct_change = (current_long - previous_long) / previous_long * 100
                short_pct_change = (current_short - previous_short) / previous_short * 100
                
                current_ratio = current_long / current_short
                previous_ratio = previous_long / previous_short
                ratio_change = current_ratio - previous_ratio
                
                # Add to the summary DataFrame
                change_summary.loc[client, 'Long_Pct_Change'] = long_pct_change
                change_summary.loc[client, 'Short_Pct_Change'] = short_pct_change
                change_summary.loc[client, 'Current_Ratio'] = current_ratio
                change_summary.loc[client, 'Ratio_Change'] = ratio_change

            # Display metrics in a more visual way using columns
            for client in participant_types:
                st.markdown(f"### {client}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Long Positions % Change", 
                        f"{change_summary.loc[client, 'Long_Pct_Change']:.2f}%", 
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Short Positions % Change", 
                        f"{change_summary.loc[client, 'Short_Pct_Change']:.2f}%", 
                        delta=None
                    )
                with col3:
                    st.metric(
                        "Long/Short Ratio", 
                        f"{change_summary.loc[client, 'Current_Ratio']:.2f}", 
                        f"{change_summary.loc[client, 'Ratio_Change']:+.2f}"
                    )
                
                # Add a sentiment indicator based on changes
                if change_summary.loc[client, 'Long_Pct_Change'] > change_summary.loc[client, 'Short_Pct_Change']:
                    st.markdown(f"**{client} Sentiment:** 🟢 Becoming more bullish")
                elif change_summary.loc[client, 'Long_Pct_Change'] < change_summary.loc[client, 'Short_Pct_Change']:
                    st.markdown(f"**{client} Sentiment:** 🔴 Becoming more bearish")
                else:
                    st.markdown(f"**{client} Sentiment:** ⚪ Neutral")
                
                st.markdown("---")

            # Add a visualization to compare participant behavior
            st.subheader("Participant Sentiment Comparison")

            # Prepare data for visualization
            sentiment_data = pd.DataFrame({
                'Participant': participant_types,
                'Long_Change': [change_summary.loc[client, 'Long_Pct_Change'] for client in participant_types],
                'Short_Change': [change_summary.loc[client, 'Short_Pct_Change'] for client in participant_types],
                'Net_Sentiment': [change_summary.loc[client, 'Long_Pct_Change'] - change_summary.loc[client, 'Short_Pct_Change'] 
                                for client in participant_types]
            })

            # Create a bar chart showing net sentiment
            fig_sentiment = px.bar(
                sentiment_data,
                x='Participant',
                y='Net_Sentiment',
                title="Net Position Change (Long % Change - Short % Change)",
                color='Net_Sentiment',
                color_continuous_scale=px.colors.diverging.RdBu,
                labels={'Net_Sentiment': 'Long-Short Differential (%)'}
            )

            fig_sentiment.update_layout(
                xaxis_title="Participant Type",
                yaxis_title="Long vs Short Change Differential (%)",
                coloraxis_colorbar=dict(title="Sentiment")
            )

            fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray", 
                                annotation_text="Neutral Line")

            st.plotly_chart(fig_sentiment, use_container_width=True)

            # Add interpretation guide
            st.info("""
            ### Interpreting Participant Sentiment:
            - **Positive values (blue)**: Participant is adding more long positions than short positions (becoming more bullish)
            - **Negative values (red)**: Participant is adding more short positions than long positions (becoming more bearish)
            - **Values near zero**: Participant is maintaining similar position balance

            Compare different participants to understand which market players are driving current trends.
            """)
        
        with tab3:
            st.subheader("F&O Participant Trends Over Time")
            
            # Filter data for the chart
            time_data = oi_df[oi_df['Client_Type'] != 'TOTAL'].copy()
            # Convert date to proper datetime
            time_data['Date'] = pd.to_datetime(time_data['Date'])
            
            # Position type selector
            position_type = st.selectbox(
                "Select Position Type",
                options=[
                    "Total_Long_Contracts", "Total_Short_Contracts",
                    "Future_Index_Long", "Future_Index_Short",
                    "Future_Stock_Long", "Future_Stock_Short",
                    "Option_Index_Call_Long", "Option_Index_Call_Short",
                    "Option_Index_Put_Long", "Option_Index_Put_Short",
                    "Option_Stock_Call_Long", "Option_Stock_Call_Short",
                    "Option_Stock_Put_Long", "Option_Stock_Put_Short"
                ],
                index=0
            )
            
            # Create time series chart
            fig_time = px.line(
                time_data,
                x='Date',
                y=position_type,
                color='Client_Type',
                title=f"{position_type.replace('_', ' ')} Trend",
                labels={'Date': 'Trading Date', position_type: 'Contracts'},
                markers=True
            )
            fig_time.update_layout(xaxis_title="Date", yaxis_title="Contracts")
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Show stacked area chart
            st.subheader("Composition of Market Participants Over Time")
            
            fig_area = px.area(
                time_data,
                x='Date',
                y=position_type,
                color='Client_Type',
                title=f"{position_type.replace('_', ' ')} Distribution",
                labels={'Date': 'Trading Date', position_type: 'Contracts'}
            )
            st.plotly_chart(fig_area, use_container_width=True)
        
        with tab4:
            st.subheader("Long-Short Ratio Analysis")
            
            # Calculate long-short ratio for each client type
            ratio_data = oi_df.copy()
            ratio_data['Long_Short_Ratio'] = ratio_data['Total_Long_Contracts'] / ratio_data['Total_Short_Contracts']
            
            # Filter out TOTAL row and NaN/inf values
            ratio_data = ratio_data[ratio_data['Client_Type'] != 'TOTAL']
            ratio_data = ratio_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Long_Short_Ratio'])
            
            # Convert date to proper datetime for sorting
            ratio_data['Date'] = pd.to_datetime(ratio_data['Date'])
            ratio_data = ratio_data.sort_values('Date')
            
            # Create long-short ratio chart
            fig_ratio = px.line(
                ratio_data,
                x='Date',
                y='Long_Short_Ratio',
                color='Client_Type',
                title="Long-Short Ratio by Participant Type",
                labels={'Date': 'Trading Date', 'Long_Short_Ratio': 'Long/Short Ratio'},
                markers=True
            )
            
            # Add reference line at 1.0
            fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                              annotation_text="Equal Long-Short")
            
            fig_ratio.update_layout(
                yaxis_title="Long/Short Ratio (> 1.0 means more long positions)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            # Show explanation
            st.info("""
            ### Understanding the Long-Short Ratio:
            - **Ratio > 1.0**: Participant holds more long positions than short positions (bullish)
            - **Ratio = 1.0**: Participant has equal long and short positions (neutral)
            - **Ratio < 1.0**: Participant holds more short positions than long positions (bearish)
            
            The ratio trends can be indicative of market sentiment among different participant types.
            """)

# Add a refresh button for real-time updates
if st.button("🔄 Refresh Alerts"):
    st.rerun()
