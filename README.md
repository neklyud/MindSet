1. git clone https://github.com/neklyud/MindSet
2. cd MindSet
3. docker-compose up --build
4. run in browser localhost:5000/upload and load your test.csv
5. send this json (i am use util Postman):
{
	"path_to_test_csv": "./Downloads/test.csv",
	"path_to_onehotencoder": "./models/ohe.pkl",
	"path_to_model": "./models/ridge.pkl",
	"path_to_poly": "./models/poly.pkl",
	"path_to_scaler": "./models/scl.pkl"
}
