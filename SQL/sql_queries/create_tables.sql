-- Creating the heart_disease_risk table
CREATE TABLE heart_disease_risk (
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

-- Creating the diagnoses table
CREATE TABLE diagnoses (
    id SERIAL PRIMARY KEY,
    diagnosis TEXT
);
