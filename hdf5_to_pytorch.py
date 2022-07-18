#hdf5 model to pytorch

import sys
import transformers # 3.0.2
from transformers import AutoModelForSequenceClassification


def main():
	predictor_path = sys.argv[1]
	pt_path = sys.argv[2]
	AutoModelForSequenceClassification.from_pretrained(predictor_path, 
                                                  from_tf=True).save_pretrained(pt_path)
	print("hdf5 saved as pytorch model succesfully to path -->",sys.argv[2])


if __name__ == '__main__':
    main()