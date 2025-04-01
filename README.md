# Health Prediction Using Data Mining

This project analyzes heart disease data using data mining techniques like Apriori and MinHash to uncover patterns and similarities between patients.

## UCI Health Disease Dataset

- 303 patients
- 14 health-related features

## Algorithms Used

### Standard Apriori
- Found 94 frequent patterns
- Generated 100 rules
- Example: "If chest pain type 2 and female, then likely heart disease"

### Personal Enhanced Apriori
- Lowered support threshold, raised confidence
- Found 261 patterns and 196 rules
- More detailed and clinically relevant findings

### Standard MinHash
- Detected 5,933 similar patient pairs (50%+ similarity)

### Personal Enhanced MinHash with LSH
- Improved to 6,935 similar pairs using buckets
- Faster and more accurate similarity detection

## Key Results

- Most common pattern: males without heart disease (37.6% support)
- Top rule: "If normal blood pressure and no heart disease, then likely male" (97.3% confidence)

## Visualizations

<b>Frequent patterns chart<b>
<img width="1196" alt="Screenshot 2025-04-01 at 1 40 19 AM" src="https://github.com/user-attachments/assets/2113d1f3-0da8-409c-8121-ad84a74c7be9" />
<br>
<br>
<b>Association rules network<b>
<img width="723" alt="Screenshot 2025-04-01 at 1 41 54 AM" src="https://github.com/user-attachments/assets/8ed2f95f-95c8-4dc3-a73b-9e3189327b48" />
<br>
<br>
<b>Patient similarity network<b>

<img width="699" alt="Screenshot 2025-04-01 at 1 42 20 AM" src="https://github.com/user-attachments/assets/fc0b8739-9c89-461d-a6c4-a0126562ebb6" />
<br>
<br>
<b>Results and time comparison graphs<b>
<img width="998" alt="Screenshot 2025-04-01 at 1 42 35 AM" src="https://github.com/user-attachments/assets/127067df-67d3-4c07-be98-25b37f9b2cf9" />
<img width="994" alt="Screenshot 2025-04-01 at 1 42 55 AM" src="https://github.com/user-attachments/assets/d3c84a8a-3466-41a3-bacd-2da14409fe0f" />
<br>

## Personal Enhancement Contributions

- Dynamic support in Apriori
- Confidence boosting for key medical rules
- Adaptive LSH band sizing in MinHash
- Feature weighting based on clinical importance
