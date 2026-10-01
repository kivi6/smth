# Dviračių paklausos prognozė | Intelektualiosios sistemos

Kitos valandos dviračių nuomų skaičiaus (`cnt`) prognozė, naudojant **UCI Bike Sharing Dataset**. Palyginami paprastas **Baseline** ir trys intelektualieji metodai: **CatBoost**, **MLP** ir **SVR**. Pagrindinis palyginimas pateiktas viename atkuriamame Jupyter Notebook.

**[Galutinė PDF ataskaita](docs/Galutine_ataskaita.pdf)** · **[Pagrindinis eksperimentas](notebooks/05_Modeliu_palyginimas.ipynb)** · **[AI naudojimo žurnalas](docs/AI_naudojimo_zurnalas.txt)** · **[Šaltiniai](docs/SALTINIAI.txt)**

## Tyrimo uždavinys

Remiantis kalendoriniais, prognozavimo metu prieinamais meteorologiniais ir ankstesnės paklausos požymiais, prognozuojamas **kitos valandos bendras nuomų skaičius**. `casual` ir `registered` nenaudojami, nes jų suma lygi prognozuojamam `cnt`. Būsimos tikslinės valandos faktiniai orai nenaudojami.

## Metodika

- Duomenys rikiuojami pagal tikslų laiką; `lag` požymiai jungiami pagal `datetime`, ne pagal eilutės numerį.
- Paskutinės 30 dienų atidedamos **TEST**; ankstesni duomenys sudaro **DEV**.
- DEV atliekami **3 expanding rolling-origin** patikrinimai; kiekviename mokoma tik iš ankstesnio laikotarpio.
- Visi metodai vertinami tais pačiais skaidymais ir metrikomis: **MAE, RMSE, Peak underestimation rate**. Piko slenkstis – mokymo dalies 90-asis procentilis.
- Hiperparametrų pasirinkimas remiasi DEV; galutinis TEST nenaudojamas jiems parinkti.
- Papildomai: CatBoost abliacija be istorinių `lag/rolling` požymių (DEV), meteorologinių požymių triukšmo bandymas (TEST) ir 10 didžiausių CatBoost TEST klaidų.

**CatBoost formulės ryšys su kodu:** `F(x) = F₀(x) + Σ η·hₘ(x)`. Funkcijoje `catboost_factory()` parametras `iterations=500` atitinka medžių skaičių, `learning_rate=0.05` – kiekvieno naujo medžio indėlio mastelį, o `depth=7` – medžių gylį.

## Rezultatai

Toliau – **išsaugoto pagrindinio eksperimento** (`depth=7`, `iterations=500`, `SEED=42`) rezultatai. Jei keičiami hiperparametrai, rezultatus ir grafikus būtina perskaičiuoti. Mažesnės visų trijų metrikų reikšmės yra geresnės.

### DEV rolling-origin (trijų patikrinimų vidurkiai)

| Metodas | MAE ↓ | RMSE ↓ | Piko neįvertinimas ↓ |
|---|---:|---:|---:|
| Baseline | 54.299 | 87.825 | 0.661 |
| CatBoost | 24.341 | 36.543 | 0.581 |
| MLP | 38.865 | 57.314 | 0.681 |
| SVR | 33.565 | 50.454 | 0.726 |

![DEV MAE ir RMSE](results/dev_mae_rmse.png)

### Galutinis TEST (paskutinės 30 dienų)

| Metodas | MAE ↓ | RMSE ↓ | Piko neįvertinimas ↓ |
|---|---:|---:|---:|
| Baseline | 51.302 | 81.164 | 0.780 |
| CatBoost | 20.523 | 31.373 | 0.700 |
| MLP | 29.387 | 44.141 | 0.660 |
| SVR | 25.393 | 40.071 | 0.740 |

![TEST MAE ir RMSE](results/test_mae_rmse.png)

![Faktinė ir prognozuota paklausa](results/test_30_dienu_ir_5_dienos.png)

**Interpretacija.** CatBoost turėjo mažiausią DEV ir TEST RMSE bei MAE, tačiau TEST rinkinyje MLP rečiau neįvertino paklausos pikų. Tai nėra universali modelių kokybės išvada – ji galioja šiam duomenų rinkiniui ir eksperimento sąlygoms.

### Abliacija ir atsparumas

Pašalinus istorinius `lag/rolling` požymius CatBoost DEV MAE padidėjo nuo **24.341** iki **38.843**. Atskirame TEST atsparumo bandyme į meteorologinius požymius įterptas nedidelis triukšmas; vienos triukšmo realizacijos rezultatų nereikėtų laikyti universaliu modelio stabilumo įrodymu.

![Abliacija](results/abliacija.png)

![Atsparumas](results/atsparumas.png)

## Paleidimas (Windows / VS Code / Git Bash)

**1.** Reikia įdiegto Python 3.11 ir Git Bash arba VS Code terminalo. Atsisiųskite šią repozitoriją ir atidarykite jos pagrindinį katalogą.

**2.** Git Bash terminale sukurkite virtualią aplinką ir įdiekite priklausomybes:

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Jei `py` komanda nepasiekiama, naudokite `python -m venv .venv` su įdiegtu Python 3.11. Linux / macOS aktyvavimas: `source .venv/bin/activate`.

**3. Viena komanda pagrindiniam eksperimentui pakartoti:**

```bash
python run_main.py
```

Jeigu `hour.csv` nėra projekto kataloge, scenarijus pirmiausia bandys atsisiųsti UCI archyvą. Jei automatinis atsisiuntimas nepavyktų, atsisiųskite [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), išskleiskite archyvą ir įdėkite `hour.csv` į projekto pagrindinį katalogą; pakartokite tą pačią komandą.

**Rezultatas:** `results/05_Modeliu_palyginimas_EXECUTED.ipynb` su išsaugotomis lentelėmis ir grafikais. Šaltinio Notebook nekeičiamas. Vykdymas gali užtrukti, nes atliekami keli modelių mokymai, abliacija ir atsparumo bandymas.

Norint atskirai atidaryti Notebook: `jupyter notebook` ir pasirinkti failą iš `notebooks/` katalogo. Atskiri `02`–`04` Notebook skirti kiekvieno metodo detalesnei analizei.

## Projekto struktūra

```text
IS_egzaminas/
├── README.md
├── requirements.txt
├── run_main.py
├── .gitignore
├── notebooks/
│   ├── 02_CatBoost.ipynb
│   ├── 03_MLP.ipynb
│   ├── 04_SVR.ipynb
│   └── 05_Modeliu_palyginimas.ipynb
├── results/
│   ├── dev_mae_rmse.png
│   ├── test_mae_rmse.png
│   ├── test_30_dienu_ir_5_dienos.png
│   ├── abliacija.png
│   └── atsparumas.png
└── docs/
    ├── Galutine_ataskaita.pdf
    ├── AI_naudojimo_zurnalas.txt
    └── SALTINIAI.txt
```

## Ribotumai

Duomenys apima istorinį vienos dviračių dalijimosi sistemos laikotarpį; kito miesto ar naujesnio laikotarpio tikslumas turi būti patikrintas atskirai. Modelis prognozuoja bendrą nuomų skaičių, o ne konkrečių stočių poreikį ar perskirstymo maršrutus. Realiam diegimui reikėtų naujų duomenų, orų prognozių ir papildomo perskirstymo optimizavimo.

**Duomenų šaltinis:** UCI Bike Sharing Dataset, DOI: [10.24432/C5W894](https://doi.org/10.24432/C5W894). Kiti moksliniai šaltiniai – [`docs/SALTINIAI.txt`](docs/SALTINIAI.txt).
