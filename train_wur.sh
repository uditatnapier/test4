#/bin/bash

#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species1_split0.npz -s 1 -e 100 -b 16 -d --model val_models/cvppp.0.npz --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species1_split1.npz -s 1 -e 100 -b 16 -d --model val_models/cvppp.1.npz --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species1_split2.npz -s 1 -e 100 -b 16 -d --model val_models/cvppp.2.npz --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species1_split3.npz -s 1 -e 100 -b 16 -d --model val_models/cvppp.3.npz --train --test

#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species2_split0.npz -s 2 -e 100 -b 16 -d --model val_models/cvppp.0.npz --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species2_split1.npz -s 2 -e 100 -b 16 -d --model val_models/cvppp.1.npz --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species2_split2.npz -s 2 -e 100 -b 16 -d --model val_models/cvppp.2.npz --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species2_split3.npz -s 2 -e 100 -b 16 -d --model val_models/cvppp.3.npz --train --test

#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species_all_split0.npz -s all -e 100 -b 16 -d --model val_models/cvppp.0.npz --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species_all_split1.npz -s all -e 100 -b 16 -d --model val_models/cvppp.1.npz --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species_all_split2.npz -s all -e 100 -b 16 -d --model val_models/cvppp.2.npz --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species_all_split3.npz -s all -e 100 -b 16 -d --model val_models/cvppp.3.npz --train --test


#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species1_split0.npz -s 2 -e 100 -b 16 -d -o wur_models/wur_species_train_1_test_2.0.xlsx --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species1_split1.npz -s 2 -e 100 -b 16 -d -o wur_models/wur_species_train_1_test_2.1.xlsx --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species1_split2.npz -s 2 -e 100 -b 16 -d -o wur_models/wur_species_train_1_test_2.2.xlsx --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species1_split3.npz -s 2 -e 100 -b 16 -d -o wur_models/wur_species_train_1_test_2.3.xlsx --test

#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species2_split0.npz -s 1 -e 100 -b 16 -d -o wur_models/wur_species_train_2_test_1.0.xlsx --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species2_split1.npz -s 1 -e 100 -b 16 -d -o wur_models/wur_species_train_2_test_1.1.xlsx --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species2_split2.npz -s 1 -e 100 -b 16 -d -o wur_models/wur_species_train_2_test_1.2.xlsx --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species2_split3.npz -s 1 -e 100 -b 16 -d -o wur_models/wur_species_train_2_test_1.3.xlsx --test

#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species1_scratch.npz -s 1 -e 100 -b 16 -d --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_species2_scratch.npz -s 2 -e 100 -b 16 -d --train --test
#python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" -o wur_models/wur_speciesall_scratch.npz -s all -e 100 -b 16 -d --train --test

python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species1_scratch.npz -s 2 -b 16 -d --test -o wur_models/wur_scratch_12.xlsx
python ./wur_main.py "/media/PHDDATA/Data/Plant Phenotyping/Wageningen Data" --model wur_models/wur_species2_scratch.npz -s 1 -b 16 -d --test -o wur_models/wur_scratch_21.xlsx