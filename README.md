# Laborator 3 — Rețea Generativă Adversarială (GAN) pe MNIST

Implementare și antrenare de la zero a unui GAN bazat exclusiv pe straturi `Dense`,
pentru generarea de cifre scrise de mână.

---

## 1. Configurația modelelor

Generatorul și Discriminatorul sunt **aproximativ simetrice**: unul expandează
100 → 784, celălalt comprimă 784 → 1.

| | Generator | Discriminator |
|---|---|---|
| Intrare | vector latent `z` (100) | imagine aplatizată (784) |
| Strat 1 | Dense(256) + LeakyReLU(0.2) | Dense(512) + LeakyReLU(0.2) + Dropout(0.3) |
| Strat 2 | Dense(512) + LeakyReLU(0.2) | Dense(256) + LeakyReLU(0.2) + Dropout(0.3) |
| Strat 3 | Dense(1024) + LeakyReLU(0.2) | — |
| Ieșire | Dense(784), `tanh` → [-1,1] | Dense(1), `sigmoid` → probabilitate |
| Parametri | 1.486.352 | 533.505 |

**Model combinat (GAN):**

```
z (100) → Generator → imagine (784) → Discriminator [înghețat] → probabilitate (1)
```

Generatorul are capacitate de ~3× față de Discriminator: modelarea unei distribuții
în 784 de dimensiuni e o sarcină mai grea decât o clasificare binară.

Discriminatorul apare în două contexte, cu rol diferit:

| Context | `trainable` | Ce se actualizează |
|---|---|---|
| `discriminator.train_on_batch(...)` | `True` | ponderile lui D |
| `gan.train_on_batch(...)` | `False` | ponderile lui G |

---

## 2. Justificarea hiperparametrilor

| Alegere | Valoare | Justificare |
|---|---|---|
| `LATENT_DIM` | 100 | Standard pentru MNIST. Prea mic → diversitate redusă; prea mare → spațiu latent slab acoperit. |
| `BATCH_SIZE` | 128 | 546 pași/epocă. Compromis stabilitate gradient ↔ viteză. Batch-uri mici amplifică oscilațiile adversariale. |
| `EPOCHS` | 30 | ≈ 16.400 actualizări per rețea. |
| `learning_rate` | 2e-4, **egal la ambele** | Valoare consacrată de DCGAN. Rate inegale rup echilibrul jocului. |
| `beta_1` (Adam) | 0.5 | Implicitul 0.9 păstrează prea mult din direcțiile trecute, iar în GAN ținta se mișcă la fiecare pas. |
| `LeakyReLU`, nu `ReLU` | slope 0.2 | `ReLU` anulează gradientul pe zona negativă. Semnalul lui G vine indirect, prin D — pierderea lui e costisitoare. |
| `Dropout(0.3)` | **doar în D** | Slăbește deliberat Discriminatorul. Un D perfect dă `D(G(z)) ≈ 0` și anulează gradientul lui G. |
| Activare finală G | `tanh` | Datele sunt normalizate în [-1,1]; ieșirea trebuie pe aceeași scară, altfel D separă trivial după domeniul valorilor. |
| Activare finală D | `sigmoid` | Ieșirea trebuie interpretată ca probabilitate în (0,1). |
| Optimizator | Adam | Rate adaptive per parametru; SGD e mult mai instabil în regim adversarial. |
| Date | train + test = 70.000 | Etichetele sunt inutile pentru un GAN necondiționat; nu există „set de validare" în sens clasic. |

Learning rate, optimizator și `beta_1` sunt **identice între G și D**. Orice asimetrie
ar fi făcut una dintre rețele să domine, iar rezultatul ar fi reflectat reglajul,
nu arhitectura.

---

## 3. Funcțiile de pierdere

Ambele rețele folosesc **Binary Crossentropy** — alegerea corectă statistic pentru o
ieșire `sigmoid` (log-likelihood negativ Bernoulli). Combinația sigmoid + BCE anulează
algebric derivata sigmoidului, evitând saturarea gradientului.

### Discriminator

```
L_D = -½ [ log D(x_real) + log(1 - D(G(z))) ]
```

Minim când D atribuie `1` imaginilor reale și `0` celor generate. În cod: două apeluri
`train_on_batch` (real→1, fals→0), mediate.

### Generator — varianta *non-saturating*

```
L_G = -log D(G(z))
```

Implementat prin `gan.train_on_batch(noise, y_real)` — imaginile generate primesc
eticheta `1`. Nu este o eroare: în loc să minimizăm `log(1 - D(G(z)))`, care dă gradient
aproape nul când G e încă slab, maximizăm `log D(G(z))`. Aceeași funcție BCE, țintă
inversată, gradient mult mai puternic la început.

### Valoarea de referință: ln 2 ≈ 0,693

La echilibrul teoretic `D(x) = 0.5` pentru orice intrare, iar ambele pierderi devin
`ln 2`. Aceasta este valoarea față de care se citesc toate rezultatele de mai jos.

> **Esențial:** cele două pierderi sunt antagoniste. Prin construcție, nu pot scădea
> simultan. Spre deosebire de un autoencoder, unde loss-ul măsoară direct calitatea
> reconstrucției, aici pierderile măsoară doar **performanța relativă** față de un
> adversar care se schimbă continuu.

---

## 4. Monitorizarea antrenării

| Semnal | Interpretare |
|---|---|
| D loss ≈ 0,5–0,7 · D acc ≈ 50–80 % | Echilibru sănătos — regimul dorit |
| D acc → 100 % · G loss explodează | D prea puternic; G rămâne fără gradient |
| D loss → 0 | D a câștigat definitiv; antrenarea a stagnat |
| G loss scade monoton | **Suspect** — de regulă semnalează colapsul lui D, nu imagini mai bune |
| Oscilații moderate | Normale și așteptate |

Într-o rețea de clasificare obișnuită funcția obiectiv e fixă, deci loss descrescător
= progres. Într-un GAN, obiectivul lui G depinde de parametrii lui D și invers —
ținta se mișcă. **Pierderile sunt instrument de diagnostic, nu măsură a calității.**

---

## 5. Rezultate obținute

Ultimele 5 epoci (CPU, ≈ 45,7 s/epocă, ≈ 19,6 min total):

| Epocă | D loss | D acc | G loss | Timp cumulat |
|---:|---:|---:|---:|---:|
| 26 | 0,6828 | 55,9 % | 0,7879 | 995,7 s |
| 27 | 0,6835 | 55,5 % | 0,7862 | 1040,6 s |
| 28 | 0,6839 | 55,4 % | 0,7812 | 1085,9 s |
| 29 | 0,6838 | 55,6 % | 0,7806 | 1131,5 s |
| 30 | 0,6833 | 55,6 % | 0,7802 | 1178,4 s |

### 5.1 Regimul este cel de echilibru

D loss ≈ 0,683 stă chiar **sub** `ln 2 = 0,693`, iar acuratețea de 55,6 % este doar
marginal peste hazard (50 %). Discriminatorul are un avantaj mic, dar real — exact ce
se dorește.

Cele trei metrici sunt coerente între ele:

```
D acc ușor peste 50 %  ↔  D loss ușor sub ln 2  ↔  G loss (0,780) ușor peste ln 2
```

G loss > ln 2 înseamnă că D încă înclină spre eticheta „fals" pentru imaginile generate,
dar fără să le respingă categoric.

### 5.2 Niciun mod de eșec clasic nu s-a produs

Nu apare divergență, D acc nu urcă spre 100 %, G loss nu explodează, D loss nu tinde
spre 0. Pe axa **stabilitate**, antrenarea a reușit.

### 5.3 Sistemul a atins un punct staționar

Pe 5 epoci, D loss variază în intervalul 0,6828–0,6839 (amplitudine 0,0011, ≈ 0,16 %),
iar G loss scade cu 0,008. Sunt valori practic constante. Două lecturi sunt posibile:

1. jocul a convers către un echilibru stabil — interpretarea favorabilă;
2. ambele rețele au încetat să mai progreseze, iar antrenarea suplimentară nu ar aduce câștig.

**Metricile singure nu pot distinge între cele două.** Comparația vizuală dintre
instantaneul de la epoca 25 și cel de la epoca 30 rezolvă ambiguitatea: dacă imaginile
sunt vizibil identice, este a doua situație.

### 5.4 Limitare importantă

**Mode collapse nu apare în aceste cifre.** Un Generator care produce 25 de exemplare
aproape identice poate menține exact aceleași valori de loss. Verificarea se face
exclusiv în grila 5×5 finală.

> **Precizare metodologică:** documentul se bazează pe ultimele 5 epoci. Nu se poate
> afirma nimic despre traiectoria de la epoca 1 la 25 — dacă echilibrul s-a stabilit
> devreme sau târziu, dacă au existat oscilații intermediare. Graficele complete din
> notebook conțin această informație.

---

## 6. Evaluarea calității generării

Trei instrumente, în ordinea importanței:

**1. Zgomot fix** — aceiași 25 de vectori latenți la epocile 0, 5, 10, …, 30. Cu `z`
constant, orice diferență între instantanee provine exclusiv din modificarea ponderilor
lui G, nu din alte eșantioane. Izolează variabila de interes.

**2. Grila finală 5×5** — 25 de vectori **noi**, din distribuția completă. Testează
generalizarea, nu doar cele 25 de puncte urmărite.

**3. Comparație real vs. generat** — evidențiază diferențele de claritate și netezime
a fundalului.

### Criterii de verificat pe imagini

- [ ] Câte din cele 25 de imagini sunt recognoscibile fără ezitare?
- [ ] Ce cifre lipsesc complet din grilă?
- [ ] Există forme ambigue (4/9, 3/8, 5/6)?
- [ ] Ce artefacte apar (zgomot de fundal, linii întrerupte, grosime inegală)?
- [ ] **Există duplicate cvasi-identice?** → semnal de mode collapse
- [ ] La ce epocă au apărut primele forme recognoscibile?

---

## 7. Interpretarea rezultatelor

Cifrele cu topologie simplă (`1`, `7`, `0`, `9`) apar de regulă mai des decât cele cu
bucle multiple sau intersecții (`8`, `2`, `5`) — au mai puțină variabilitate structurală
de modelat.

Artefactele caracteristice acestei arhitecturi — fundal cu „zgomot de sare", contururi
îngroșate, detalii fine pierdute — au cauză **structurală**: aplatizarea 28×28 → 784
distruge vecinătatea spațială, iar straturile `Dense` nu au invarianță la translație
sau partajare de parametri. Fiecare poziție de pixel este învățată independent, deci
netezimea locală nu poate fi impusă.

> **Regula de aur:** corelează întotdeauna graficele cu imaginile. Un G loss în scădere
> fără ameliorare vizuală înseamnă dezechilibru, nu progres. Invers, metricile de
> echilibru obținute aici sunt o condiție **necesară, nu suficientă**, pentru imagini bune.

---

## 8. Pașii proiectului

| # | Pas | Ce face |
|---:|---|---|
| 1 | Import biblioteci | Încarcă NumPy, Matplotlib, TensorFlow/Keras, tqdm. Afișează versiunea TF și disponibilitatea GPU. |
| 2 | Seed-uri | Fixează seed-urile NumPy/TF pentru reproducibilitate aproximativă (nu bit-identică între mașini). |
| 3 | Pregătire date | Încarcă MNIST, concatenează train+test (70.000 imagini), normalizează [0,255] → [-1,1] pentru potrivire cu `tanh`, aplatizează 28×28 → 784. |
| 4 | Vizualizare date | Afișează imagini reale — referință pentru comparația finală. |
| 5 | Parametri | Definește constantele: dimensiuni, batch, epoci, learning rate, frecvența instantaneelor. |
| 6 | Generator | Construiește rețeaua 100→256→512→1024→784 cu LeakyReLU și `tanh` final. |
| 7 | Discriminator | Construiește rețeaua 784→512→256→1 cu LeakyReLU, Dropout și `sigmoid` final. |
| 8 | Compilare D | Adam + BCE + metrică accuracy. Compilat separat, pentru antrenarea directă. |
| 9 | Construire GAN | Leagă G→D într-un model unic, cu `discriminator.trainable = False` **înainte** de compilare, astfel încât prin GAN să se actualizeze doar ponderile Generatorului. |
| 10 | Funcții auxiliare | Zgomot fix pentru urmărirea evoluției; construirea mozaicurilor 5×5; normalizarea metricilor returnate de `train_on_batch`. |
| 11 | Bucla de antrenare | Per pas: (a) antrenează D pe un batch real (→1) și unul generat (→0); (b) antrenează G prin GAN cu imagini false etichetate 1. Salvează metricile; instantaneu la fiecare 5 epoci. |
| 12 | Grafice pierderi | Trasează D loss, G loss, D accuracy. Diagnostic, nu certificat de calitate. |
| 13 | Evoluție vizuală | Aliniază instantaneele din zgomotul fix — transformarea din zgomot în cifre. |
| 14 | Generare finală | 25 de vectori noi → G → conversie [-1,1] → [0,1] → reshape 784 → 28×28 → grilă 5×5. |
| 15 | Analiză | Evaluare vizuală pe criteriile din secțiunea 6, corelată cu graficele. |
| 16 | Concluzie | Sinteză: implementare, roluri G/D, mecanism adversarial, rezultate, limitări. |

---

## 9. Limitări

1. **Aplatizarea distruge structura spațială.** Fără invarianță la translație și fără
   partajare de parametri — de aici contururile difuze și zgomotul de fundal.
2. **Raport slab parametri/capacitate.** ≈ 2 M parametri pentru o calitate sub cea a
   unui DCGAN cu mult mai puțini.
3. **Instabilitate potențială.** Risc de mode collapse și de dominare a Discriminatorului,
   fără mecanisme de stabilizare (BatchNorm, Wasserstein loss, gradient penalty).
4. **Fără metrică obiectivă.** Pierderile nu indică convergența; evaluarea rămâne
   vizuală și subiectivă.
5. **Model necondiționat.** Nu se poate cere o cifră anume; ar fi necesar un cGAN.

---

## 10. Direcții de îmbunătățire

În ordinea raportului beneficiu/efort:

1. Înlocuirea straturilor `Dense` cu convoluții (DCGAN: `Conv2DTranspose` în G,
   `Conv2D` cu stride în D)
2. `BatchNormalization` în Generator
3. Label smoothing (`y_real = 0.9` în loc de `1.0`) — modificare de o linie, reduce
   riscul ca D să devină prea încrezător
4. Wasserstein GAN cu gradient penalty
5. Variantă condiționată (cGAN), pentru control asupra cifrei generate
6. Evaluare cantitativă cu FID, în locul inspecției vizuale
