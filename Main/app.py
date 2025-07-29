import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from unidecode import unidecode

# Load dataset and model
data = pd.read_csv("Main/dataset.csv")
model = joblib.load("player_rating_pipeline.pkl")

# Clean player names for better matching
data['name_clean'] = data['display_name'].apply(lambda x: unidecode(x.lower()))

# Define helper function to suggest player names
def get_suggestions(input_name):
    clean_input = unidecode(input_name.lower())
    suggestions = data[data['name_clean'].str.contains(clean_input, na=False)]['display_name'].unique().tolist()
    return suggestions

# Define function to predict rating and return stats
def predict_rating(name):
    row = data[data['display_name'] == name]
    if not row.empty:
        stats = row[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']].iloc[0]
        input_df = pd.DataFrame([stats.values], columns=stats.index)
        rating = model.predict(input_df)[0]
        return rating, stats
    return None, None

# Streamlit page settings
st.set_page_config(page_title="Football Player App", layout="wide")
st.title("⚽ Football Player Performance Tool")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("📌 App Info")
    st.markdown("""
    This app allows you to:
    - Predict a football player's performance rating using key stats.
    - Compare stats of any two players visually.

    **Used Features**:
    - Pace  
    - Shooting  
    - Passing  
    - Dribbling  
    - Defending  
    - Physic  

    💡 Tip: New players can be tested by typing in custom stats!
    """)
    st.markdown("---")

# ------------------ Tabs ------------------
tab1, tab2 = st.tabs(["Predict Player Performance", "Compare Two Players"])

# --- Tab 1: Predict Player Rating ---
with tab1:
    st.header("🔢 Predict Player's Performance")

    # Player Name
    player_name = st.text_input("Enter Player Name")

    # Input sliders with unique keys
    inputs = {
        "pace": st.number_input("Pace", min_value=0.0, max_value=100.0, step=0.1, key="pace_input"),
        "shooting": st.number_input("Shooting", min_value=0.0, max_value=100.0, step=0.1, key="shooting_input"),
        "passing": st.number_input("Passing", min_value=0.0, max_value=100.0, step=0.1, key="passing_input"),
        "dribbling": st.number_input("Dribbling", min_value=0.0, max_value=100.0, step=0.1, key="dribbling_input"),
        "defending": st.number_input("Defending", min_value=0.0, max_value=100.0, step=0.1, key="defending_input"),
        "physic": st.number_input("Physical", min_value=0.0, max_value=100.0, step=0.1, key="physic_input")
    }

    # Validate inputs
    missing_fields = [field for field, value in inputs.items() if value == 0.0]
    is_name_missing = player_name.strip() == ""

    if st.button("Predict Rating"):
        if is_name_missing or missing_fields:
            st.warning("⚠️ Please fill all the required fields to proceed.")
            if is_name_missing:
                st.markdown("<p style='color:red;'>Player name is required</p>", unsafe_allow_html=True)
            for field in missing_fields:
                st.markdown(f"<p style='color:red;'>{field.capitalize()} cannot be 0.0</p>", unsafe_allow_html=True)
        else:
            # Create input DataFrame
            input_df = pd.DataFrame([{
                "pace": inputs["pace"],
                "shooting": inputs["shooting"],
                "passing": inputs["passing"],
                "dribbling": inputs["dribbling"],
                "defending": inputs["defending"],
                "physic": inputs["physic"]
            }])

            # Predict
            predicted_rating = model.predict(input_df)[0]
            st.metric(label=f"⭐ Performance of {player_name} based on the input", value=f"{predicted_rating:.2f}")

# --- Tab 2: Compare Players ---
with tab2:
    st.header("Compare Two Players")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Player 1")
        player1_input = st.text_input("Type a name:", key="p1")
        suggestions1 = get_suggestions(player1_input)[:5] if player1_input else []
        player1 = st.selectbox("Choose Player 1", options=suggestions1) if suggestions1 else None

    with col2:
        st.subheader("🔍 Player 2")
        player2_input = st.text_input("Type a name:", key="p2")
        suggestions2 = get_suggestions(player2_input)[:5] if player2_input else []
        player2 = st.selectbox("Choose Player 2", options=suggestions2) if suggestions2 else None

    if player1 and player2:
        rating1, stats1 = predict_rating(player1)
        rating2, stats2 = predict_rating(player2)

        if rating1 is not None and rating2 is not None:
            st.markdown("## 🔮 Predicted Performance Ratings")

            col1, col2 = st.columns(2)
            col1.metric(label=f"⭐ {player1}", value=round(rating1))
            col2.metric(label=f"⭐ {player2}", value=round(rating2))

            winner = player1 if rating1 > rating2 else player2 if rating2 > rating1 else "It's a Tie!"
            st.success(f"👑 **Likely Better Performer:** `{winner}`")

            # Stat Table
            st.markdown("### 📊 Stat-by-Stat Comparison")
            stat_names = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physic']
            df_stats = pd.DataFrame({
                'Stat': stat_names,
                player1: stats1.values,
                player2: stats2.values
            })
            st.dataframe(df_stats.set_index('Stat'), height=300, use_container_width=True)

            # Radar Chart - Improved Version
            st.markdown("### 🕸️ Visual Comparison: Radar Chart")

            labels = stat_names
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]

            stats1_plot = stats1.tolist() + [stats1.tolist()[0]]
            stats2_plot = stats2.tolist() + [stats2.tolist()[0]]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

            # Plot Player 1
            ax.plot(angles, stats1_plot, color='deepskyblue', linewidth=2, label=player1.split('(')[0].strip())
            ax.fill(angles, stats1_plot, color='deepskyblue', alpha=0.25)

            # Plot Player 2
            ax.plot(angles, stats2_plot, color='darkorange', linewidth=2, label=player2.split('(')[0].strip())
            ax.fill(angles, stats2_plot, color='darkorange', alpha=0.25)

            # Format chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
            ax.set_title("🎯 Player Stat Comparison Radar", size=16, pad=20)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)

            # Add background grid
            ax.grid(True)

            st.pyplot(fig)


        else:
            st.warning("⚠️ Couldn’t find stats for one or both players.")
    else:
        st.info("Type and select two players to get started!")
