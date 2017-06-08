# Get data first
echo -e "\n\nUsing python at: $(which python)\n\n"
python --version
echo -e "\n"

echo -e "\nDownloading data\n"
t1=`date +%s`

cd data
python getdata.py
cd ..

t2=`date +%s`
echo -e "\n\nDownloaded data in $((t2-t1)) s\n\n"

# Perform data workup
echo -e "pk-EFM data workup \n\n"

cd scripts

python 001-pk-efm-workup.py
# Arguments: [chains] [iterations]
python 002-pk-efm-stan.py 2 1000
python 003-pk-efm-plot.py

t1=`date +%s`
echo -e "\nWorked up pk-EFM data in $((t1-t2)) s\n\n"
echo -e "\n\ntr-EFM data workup \n\n"

python 010-tr-efm-workup.py
python 011-tr-efm-stan.py

cd ..
t2=`date +%s`
echo -e "Worked up pk-EFM data in $((t2-t1)) s\n\n"

# Generate plots
echo -e "\n\nGenerating plots\n\n"
cd figs_scripts

python 010-photocapacitance.py
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 021-AF-main-revised.ipynb
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 022-fig2-variable-frequency-lock-in.ipynb
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 023-phase-shift.ipynb
python 030-freq-phase-noise.py
python 040-pk-efm-tr-efm.py
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 055-tr-efm-initial-voltage-pulse.ipynb
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 056-time-domain-voltage-pulse.ipynb
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 060-subcycle-pk-efm-all.ipynb


t1=`date +%s`
cd ..
echo -e "Generated plots in $((t1-t2))"

