-- Adding data to the heart_disease_risk table
INSERT INTO heart_disease_risk (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, decision)
VALUES (63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0, 0);

-- Adding data to the diagnoses table
INSERT INTO diagnoses (diagnosis)
VALUES ('No risk'), ('Risk exists');

-- Inserting new data into the heart_disease_risk table with a default filename
INSERT INTO heart_disease_risk (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, decision, file_name)
VALUES (55, 1, 4, 135, 255, 0, 0, 142, 1, 3.1, 1, 0, 6, 1, 'heart_data_default.csv');

-- Inserting multiple rows of data into the heart_disease_risk table with different filenames
INSERT INTO heart_disease_risk (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, decision, file_name)
VALUES (52, 0, 3, 125, 212, 1, 1, 168, 0, 1.0, 2, 2, 7, 0, 'heart_data_1.csv'),
       (59, 1, 2, 140, 234, 0, 0, 156, 1, 0.5, 3, 1, 3, 1, 'heart_data_2.csv'),
       (44, 0, 1, 118, 242, 0, 0, 149, 0, 0.3, 1, 0, 3, 0, 'heart_data_3.csv');

-- Inserting data into the heart_disease_risk table with a filename generated based on current timestamp
INSERT INTO heart_disease_risk (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, decision, file_name)
VALUES (48, 1, 2, 130, 256, 1, 0, 150, 1, 1.5, 2, 0, 6, 1, CONCAT('heart_data_', NOW()::date, '.csv'));

-- Inserting data into the heart_disease_risk table and updating the filename later
INSERT INTO heart_disease_risk (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, decision, file_name)
VALUES (41, 0, 2, 120, 218, 0, 0, 150, 0, 1.8, 1, 0, 7, 0, 'heart_data_new.csv');

-- Later, update the filename for this specific record
UPDATE heart_disease_risk
SET file_name = 'heart_data_final.csv'
WHERE id = 10;
