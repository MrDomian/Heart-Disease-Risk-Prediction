-- Deleting data from the heart_disease_risk table
DELETE FROM diagnoses
WHERE id = 1;

-- Deleting data from the heart_disease_risk table based on the filename
DELETE FROM heart_disease_risk
WHERE file_name = 'heart_disease_risk.csv';

-- Updating the filename for a specific record in the heart_disease_risk table
UPDATE heart_disease_risk
SET file_name = 'new_heart_data.csv'
WHERE id = 3;

-- Inserting new data into the heart_disease_risk table along with the filename
INSERT INTO heart_disease_risk (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, decision, file_name)
VALUES (58, 0, 2, 130, 197, 1, 1, 131, 0, 1.2, 2, 0, 7, 1, 'heart_data_new.csv');

-- Querying data from the heart_disease_risk table based on the filename
SELECT age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, decision
FROM heart_disease_risk
WHERE file_name = 'heart_data_old.csv';
