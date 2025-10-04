# Prediction-of-Photovoltaic-Properties-ofDyes-usingDeeplearning
Designed a deep learning-based framework to predict photovoltaic dye properties — absorption maxima, LUMO energy, and bandgap — using SMILES notation, solvent data, and Mordred molecular descriptors.

Trained and compared FNN, 1D CNN, and MLP architectures; the optimized MLP achieved the lowest test MSE (334.70) and R² = 0.70, outperforming FNN (MSE = 486.65) and CNN (MSE = 596.70).

Selected the top 20 correlated descriptors from 1800 features to enhance accuracy and speed, and deployed the final MLP in a Tkinter-based GUI for real-time dye property prediction and virtual screening of DSSC candidates.
