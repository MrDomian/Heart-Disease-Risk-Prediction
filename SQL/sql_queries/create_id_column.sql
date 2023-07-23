-- Create a new table with the 'id' column in the first position
CREATE TABLE new_heart_disease_risk (
    id SERIAL PRIMARY KEY,
    age FLOAT,
    sex FLOAT,
    cp FLOAT,
    trestbps FLOAT,
    chol FLOAT,
    fbs FLOAT,
    restecg FLOAT,
    thalach FLOAT,
    exang FLOAT,
    oldpeak FLOAT,
    slope FLOAT,
    ca FLOAT,
    thal FLOAT,
    decision INTEGER
);

-- Copying data from the original table into the new table
INSERT INTO new_heart_disease_risk (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, decision)
SELECT age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, decision
FROM heart_disease_risk;

-- Deletion of the original table
DROP TABLE heart_disease_risk;

-- Renaming the new table to the original name
ALTER TABLE new_heart_disease_risk RENAME TO heart_disease_risk;
