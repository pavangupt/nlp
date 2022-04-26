gcloud builds submit --tag gcr.io/nlp-all-in-ons/nlpallio  --project=nlp-all-in-ons

gcloud run deploy --image gcr.io/nlp-all-in-ons/nlpallio --platform managed  --project=nlp-all-in-ons --allow-unauthenticated