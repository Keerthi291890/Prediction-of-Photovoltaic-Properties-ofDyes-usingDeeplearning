import streamlit as st
import numpy as np
import pickle
import joblib
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Draw
from mordred import Calculator, descriptors
from fpdf import FPDF
import tempfile
import os

# Load the trained model and preprocessing tools
model = tf.keras.models.load_model("dye_model.keras")

with open("descriptor_columns.pkl", "rb") as f:
    descriptor_columns = pickle.load(f)

scaler = joblib.load("scaler.pkl")
solvent_encoder = joblib.load("solvent_encoder.pkl")

# Mordred descriptor calculator
calc = Calculator(descriptors, ignore_3D=True)
known_solvents = list(solvent_encoder.categories_[0])  # List of known solvents

# Streamlit UI
st.set_page_config(page_title="Dye Property Predictor", layout="centered")
st.title("Dye Property Predictor")
st.markdown("Enter a **SMILES string** and choose a **solvent** to predict:")
st.markdown("- **LUMO Energy** (eV)  \n- **Band Gap** (eV)  \n- **Absorption Maxima** (nm)")

# Input fields
smiles_input = st.text_input("Enter SMILES string of the dye:")
solvent_input = st.text_input("Enter solvent:")

# Prediction logic
if st.button("Predict"):
    try:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error(" Invalid SMILES string.")
        elif solvent_input not in known_solvents:
            st.error("❌ Solvent not recognized. Please enter a known solvent.")
        else:
            st.image(Draw.MolToImage(mol, size=(300, 300)), caption="🧪 Molecule Structure")

            # Mordred descriptors
            mordred_df = calc.pandas([mol])
            mordred_df = mordred_df[descriptor_columns]
            mordred_df = mordred_df.fillna(0)

            # Scale and encode
            scaled_descriptors = scaler.transform(mordred_df)
            encoded_solvent = solvent_encoder.transform([[solvent_input]])
            input_features = np.concatenate([scaled_descriptors, encoded_solvent], axis=1)

            # Prediction
            predictions = model.predict(input_features)[0]
            lumo_energy, band_gap, absorption_maxima = predictions

            # Show predictions
            st.success("Prediction Successful!")
            st.write(f"**Absorption Maxima:** {lumo_energy:.2f} nm")
            st.write(f"**LUMO Energy:** {band_gap:.4f} eV")
            st.write(f"**Band Gap:** {absorption_maxima:.4f} eV")

            # Generate PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Dye Property Prediction Report", ln=True, align='C')
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"SMILES: {smiles_input}", ln=True)
            pdf.cell(200, 10, txt=f"Solvent: {solvent_input}", ln=True)
            pdf.cell(200, 10, txt=f"Absorption Maxima: {lumo_energy:.2f} eV", ln=True)
            pdf.cell(200, 10, txt=f"LUMO Energy:: {band_gap:.4f} eV", ln=True)
            pdf.cell(200, 10, txt=f"Band Gap: {absorption_maxima:.4f} eV", ln=True)

            # Write PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf.output(tmp_file.name, 'F')  # Write PDF to temporary file
                tmp_file_path = tmp_file.name

            # Read the temporary file into bytes
            with open(tmp_file_path, 'rb') as f:
                pdf_bytes = f.read()

            # Clean up the temporary file
            os.unlink(tmp_file_path)

            # Download button
            st.download_button(
                label="📄 Download Report as PDF",
                data=pdf_bytes,
                file_name="dye_prediction_report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
