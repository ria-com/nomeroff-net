example curl to backend 
```bash
curl --header "Content-Type: application/json" --request POST --data '
{
    "base_dir": "/mnt/datasets/anpr_ocr/ukr_data/test/anpr_ocr/test", 
    "chended_numbers":[
        {
            "number": "AA0013BM", 
            "newNumber": "AA0013BM"
        }
    ], 
    "who_changed": "I"
}'  http://127.0.0.1:5000/regionOCRModeration
```