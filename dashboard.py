import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# --- Streamlit Dashboard ---
# This is the main application file for the price optimization dashboard.

# --- Page Configuration ---
st.set_page_config(
    page_title="FreshVault Dashboard",
    page_icon="🔐",
    layout="wide"
)

# --- File Paths ---
APPROVED_FILE = "approved_items.csv"
SAMPLE_FILE = "sample_inventory.csv" # New file path
MODEL_FILE = "ml_model.pkl"
ENCODERS_FILE = "encoders.pkl"


# --- Load Model and Encoders ---
@st.cache_resource
def load_model_and_encoders():
    """Loads the ML model and encoders from disk."""
    try:
        model = joblib.load(MODEL_FILE)
        encoders = joblib.load(ENCODERS_FILE)
        return model, encoders
    except FileNotFoundError:
        return None, None


model, encoders = load_model_and_encoders()


# --- Initialize Storage ---
def init_storage():
    """Initializes the CSV file for approved items if it doesn't exist."""
    if not os.path.exists(APPROVED_FILE):
        df = pd.DataFrame(columns=[
            'item_name', 'product_type', 'days_to_expiry', 'stock_quantity',
            'selling_price', 'suggested_discount_percentage', 'discounted_price'
        ])
        df.to_csv(APPROVED_FILE, index=False)


init_storage()

# --- Global Styling (Dark Theme) ---
st.markdown("""
<style>
    /* Main background and text colors */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA;
    }
    .st-emotion-cache-16txtl3 { /* Main content area */
        color: #FAFAFA;
    }
    /* General Button Styling */
    .stButton>button {
        background-color: #00B48A; /* A vibrant green */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #009A75;
    }

    /* More specific selector for the reject button to avoid layout issues */
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) .stButton > button,
    .remove-button button {
        background-color: #D32F2F !important; /* Red for reject/remove */
    }
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) .stButton > button:hover,
    .remove-button button:hover {
        background-color: #B71C1C !important;
    }

    /* Card Styling */
    .card {
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
        background-color: #161A25;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        color: #FAFAFA;
    }
    /* Expander styling */
    .st-emotion-cache-p5msec { /* Expander header */
        background-color: #262730;
        border-radius: 8px;
    }
    /* Dataframe styling */
    .stDataFrame {
        background-color: #0E1117;
    }

    /* Hide the "Press Enter to submit" text in forms more forcefully */
    [data-testid="stForm"] small {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Handling ---
if 'role' not in st.session_state:
    st.session_state.role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'predicted_items' not in st.session_state:
    st.session_state.predicted_items = None
if 'item_actions' not in st.session_state:
    st.session_state.item_actions = {}  # To track approve/reject status


# --- Login Section ---
def login_section():
    """Displays the login form."""
    st.title("🔐 Login to FreshVault")

    with st.form("login_form"):
        username = st.text_input("Enter your name:")
        password = st.text_input("Enter any password:", type="password")
        role = st.selectbox("Select your role:", ["Store Manager", "Customer"])
        submitted = st.form_submit_button("Login")

        if submitted:
            if username and password:
                st.session_state.username = username
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Please enter a username and password.")


# --- Store Manager View ---
def manager_view():
    """Displays the dashboard for the store manager."""
    st.title("📊 Manager Dashboard")
    st.write(f"👤 Logged in as: **{st.session_state.username}**")
    
    st.markdown("---")

    # --- Section for Prediction and Approval ---
    st.header("📤 Predict & Approve New Discounts")
    
    # --- NEW: Add download button for sample CSV ---
    try:
        with open(SAMPLE_FILE, "r") as f:
            sample_csv = f.read()
        st.download_button(
            label="📄 Don't have a CSV? Download a sample file here.",
            data=sample_csv,
            file_name='sample_inventory.csv',
            mime='text/csv',
        )
    except FileNotFoundError:
        st.warning("sample_inventory.csv not found. The download link will not be available.")

    uploaded_file = st.file_uploader("Upload Inventory CSV for Prediction", type="csv", label_visibility="collapsed")

    if uploaded_file and st.session_state.predicted_items is None:
        new_items_df = pd.read_csv(uploaded_file)

        if model and encoders:
            predict_df = new_items_df.copy()
            for col, encoder in encoders.items():
                if col in predict_df.columns:
                    known_labels = encoder.classes_
                    predict_df[col] = predict_df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in known_labels else -1)

            features_order = model.get_booster().feature_names
            predict_df = predict_df.reindex(columns=features_order).fillna(0)

            predictions = model.predict(predict_df)
            new_items_df['suggested_discount_percentage'] = [round(p, 2) for p in predictions]
            new_items_df['discounted_price'] = (new_items_df['selling_price'] * (
                        1 - new_items_df['suggested_discount_percentage'] / 100)).round(2)

            st.session_state.predicted_items = new_items_df.to_dict('records')
            st.session_state.item_actions = {i: None for i in range(len(st.session_state.predicted_items))}
            st.success("Predictions generated successfully!")
            st.rerun()

    if st.session_state.predicted_items:
        predicted_df = pd.DataFrame(st.session_state.predicted_items)
        st.header("📈 Prediction Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Items Predicted", f"{len(predicted_df)}")
        avg_discount = predicted_df['suggested_discount_percentage'].mean()
        col2.metric("Avg. Discount", f"{avg_discount:.2f}%")
        top_deal = predicted_df.loc[predicted_df['suggested_discount_percentage'].idxmax()]
        col3.metric("Top Discount", f"{top_deal['item_name']} ({top_deal['suggested_discount_percentage']:.2f}%)")

        st.header("✅ Review Items & Take Action")
        for i, row in enumerate(predicted_df.to_dict('records')):
            expander_label = f"**{row['item_name']}** | Original: ${row['selling_price']:.2f} ⟶ Discounted: ${row['discounted_price']:.2f}"

            action = st.session_state.item_actions.get(i)
            if action == 'approved':
                expander_label = f"✅ {expander_label}"
            elif action == 'rejected':
                expander_label = f"❌ {expander_label}"

            with st.expander(expander_label):
                details_df = pd.DataFrame(row, index=[0]).T.rename(columns={0: 'Details'})
                details_df['Details'] = details_df['Details'].astype(str)
                st.dataframe(details_df)

                if action == 'approved':
                    st.success(f"Approved: {row['item_name']}")
                elif action == 'rejected':
                    st.error(f"Rejected: {row['item_name']}")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"Approve - {row['item_name']}", key=f"approve_{i}", use_container_width=True):
                        st.session_state.item_actions[i] = 'approved'
                        st.rerun()
                with c2:
                    if st.button(f"Reject - {row['item_name']}", key=f"reject_{i}", use_container_width=True):
                        st.session_state.item_actions[i] = 'rejected'
                        st.rerun()

        approved_indices = [i for i, action in st.session_state.item_actions.items() if action == 'approved']
        if approved_indices:
            st.header("📋 Approved Items Summary")
            approved_df_summary = predicted_df.iloc[approved_indices]
            st.dataframe(approved_df_summary)

            csv = approved_df_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Approved Items as CSV",
                data=csv,
                file_name='manager_approved_items.csv',
                mime='text/csv',
                use_container_width=True
            )

            if st.button("Confirm Final Approval", use_container_width=True):
                approved_df_to_save = approved_df_summary[
                    ['item_name', 'product_type', 'days_to_expiry', 'stock_quantity', 'selling_price',
                     'suggested_discount_percentage', 'discounted_price']]

                current_approved = pd.read_csv(APPROVED_FILE)
                updated_approved = pd.concat([current_approved, approved_df_to_save], ignore_index=True)
                updated_approved.to_csv(APPROVED_FILE, index=False)

                st.success("Final approval confirmed. Approved items are now ready for customer view!")
                st.session_state.predicted_items = None
                st.session_state.item_actions = {}
                st.balloons()

        if st.button("Clear Predictions", use_container_width=True):
            st.session_state.predicted_items = None
            st.session_state.item_actions = {}
            st.rerun()

    st.markdown("---")

    # --- Section for Managing Active Discounts ---
    st.header("🛠️ Manage Active Discounts")
    try:
        active_discounts_df = pd.read_csv(APPROVED_FILE)
        if active_discounts_df.empty:
            st.info("There are currently no active discounts to manage.")
        else:
            st.write("Here you can view and remove currently active discounts.")
            
            for index, row in active_discounts_df.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{row['item_name']}** - {row['suggested_discount_percentage']}% off (Expires in {row['days_to_expiry']} days)")
                with col2:
                    st.markdown('<div class="remove-button">', unsafe_allow_html=True)
                    if st.button(f"Remove", key=f"remove_{index}", use_container_width=True):
                        active_discounts_df.drop(index, inplace=True)
                        active_discounts_df.to_csv(APPROVED_FILE, index=False)
                        st.success(f"Removed '{row['item_name']}' from the active discounts.")
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    except pd.errors.EmptyDataError:
        st.info("There are currently no active discounts to manage.")
    except FileNotFoundError:
        st.error("approved_items.csv not found. Please approve some items first.")


# --- Customer View ---
def customer_view():
    """Displays the dashboard for the customer."""
    st.title("🛍️ FreshVault Deals")
    st.write(f"👋 Welcome, **{st.session_state.username}**! Explore exclusive discounts.")

    try:
        approved_df = pd.read_csv(APPROVED_FILE)
        approved_df = approved_df[approved_df['days_to_expiry'] > 0]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.warning("No approved items available yet. Check back soon!")
        return

    if approved_df.empty:
        st.info("No discounts are available at the moment. Please check back later!")
    else:
        st.header("📊 Deal Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Items on Sale", len(approved_df))
        col2.metric("Average Discount", f"{approved_df['suggested_discount_percentage'].mean():.1f}%")
        best_deal = approved_df.loc[approved_df['suggested_discount_percentage'].idxmax()]
        col3.metric("Best Deal", f"{best_deal['item_name']} ({best_deal['suggested_discount_percentage']}% off)")

        st.header("🔍 Filter by Category")
        product_types = ["All"] + sorted(approved_df['product_type'].unique().tolist())
        selected_type = st.selectbox("Select type:", product_types, label_visibility="collapsed")

        filtered_df = approved_df[
            approved_df['product_type'] == selected_type] if selected_type != "All" else approved_df

        st.header("🛒 Available Discounts")
        for _, row in filtered_df.iterrows():
            st.markdown(f"""
            <div class="card">
                <h4>{row['item_name']} <span style="float:right; color: #00B48A; font-size: 1.2em;">${row['discounted_price']:.2f}</span></h4>
                <p><b>Type:</b> {row['product_type']} | <b>Expires in:</b> {row['days_to_expiry']} days<br>
                   <b>Original Price:</b> <s style="color: grey;">${row['selling_price']:.2f}</s> | <b>Discount:</b> {row['suggested_discount_percentage']}% off</p>
            </div>
            """, unsafe_allow_html=True)

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Deals as CSV",
            data=csv,
            file_name='freshvault_deals.csv',
            mime='text/csv',
            use_container_width=True
        )


# --- Main App Logic ---
if st.session_state.role is None:
    login_section()
else:
    st.sidebar.title(f"Logged in as {st.session_state.role}")
    st.sidebar.write(f"User: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.session_state.role == "Store Manager":
        if not model or not encoders:
            st.error("Machine learning model or encoders not found. Please run `model.py` to generate them.")
        else:
            manager_view()
    elif st.session_state.role == "Customer":
        customer_view()
