//+------------------------------------------------------------------+
//|                                                  NFP_Neon_Pro.mq5    |
//|                               Copyright 2026, Azzo Trade         |
//|                                  https://azzogaming.netlify.app/ |
//+------------------------------------------------------------------+
#property copyright "Azzo Trade 2026"
#property link      "https://azzotrade.netlify.app"
#property version   "32.0" // F.FACTORY panelidagi sana/kun/vaqt qatori QIZILDAN CYAN rangga o'zgartirildi (default)
#property description "Bismillah, ey Allohim menga kuch va ishonch bergin, savdolarimda omad va zor bolishni nasib etgin!"
#include <Trade\Trade.mqh>
CTrade trade;

//+------------------------------------------------------------------+
//| AZZO ML MODULE: HMM (Hidden Markov Model) BOZOR-REJIM ANIQLAGICH  |
//| ----------------------------------------------------------------- |
//| Bozor har doim 3 ta "yashirin holat"dan birida deb faraz          |
//| qilinadi: TREND / RANGE / CHAOTIC. Bu holatlar to'g'ridan-to'g'ri  |
//| ko'rinmaydi, lekin ularning "izlari" - ADX (trend kuchi) va ATR    |
//| (volatillik) - ko'rinadi. Scaled Baum-Welch (EM) algoritmi tarixiy |
//| ma'lumotlardan shu 3 holatning statistik xususiyatlarini (o'rtacha,|
//| dispersiya, o'tish ehtimoli) o'zi o'rganib oladi. Viterbi          |
//| algoritmi esa hozirgi vaqtda qaysi holatda ekanligimizni aniqlaydi.|
//|                                                                     |
//| MUHIM: bu modul faqat PANELGA "YO'NALISH:" yozuvini chiqarish      |
//| uchun ishlatiladi. Robotning savdo logikasiga (NFP straddle,       |
//| grid, OCO, trailing va h.k.) HECH QANDAY ta'sir qilmaydi.          |
//+------------------------------------------------------------------+
#define HMM_NSTATES 3
#define HMM_NFEAT   2

class CHMMRegime
  {
private:
   int      m_T;
   double   m_obs[][HMM_NFEAT];
   double   m_A[HMM_NSTATES][HMM_NSTATES];
   double   m_mean[HMM_NSTATES][HMM_NFEAT];
   double   m_var[HMM_NSTATES][HMM_NFEAT];
   double   m_pi[HMM_NSTATES];
   int      m_trendState, m_rangeState, m_chaosState;
   bool     m_trained;

   double GaussianPDF(int state,double f0,double f1)
     {
      double v0=MathMax(m_var[state][0],1e-6);
      double v1=MathMax(m_var[state][1],1e-6);
      double e =-0.5*(MathPow(f0-m_mean[state][0],2)/v0+MathPow(f1-m_mean[state][1],2)/v1);
      double d = 2.0*M_PI*MathSqrt(v0*v1);
      double val=MathExp(e)/d;
      if(val<1e-300) val=1e-300;
      return val;
     }

   void LabelStates()
     {
      int best=0; double bestVal=m_mean[0][0];
      for(int s=1;s<HMM_NSTATES;s++)
         if(m_mean[s][0]>bestVal){ bestVal=m_mean[s][0]; best=s; }
      m_trendState=best;

      int rem[2]; int k=0;
      for(int s=0;s<HMM_NSTATES;s++) if(s!=best) rem[k++]=s;

      if(m_mean[rem[0]][1]<=m_mean[rem[1]][1]){ m_rangeState=rem[0]; m_chaosState=rem[1]; }
      else                                     { m_rangeState=rem[1]; m_chaosState=rem[0]; }
     }

public:
   CHMMRegime(){ Init(); }

   void Init()
     {
      m_T=0; m_trained=false;
      for(int i=0;i<HMM_NSTATES;i++)
        {
         m_pi[i]=1.0/HMM_NSTATES;
         for(int j=0;j<HMM_NSTATES;j++) m_A[i][j]=(i==j)?0.8:0.1;
         for(int f=0;f<HMM_NFEAT;f++){ m_mean[i][f]=(double)(i*7+f*3); m_var[i][f]=1.0; }
        }
      m_trendState=0; m_rangeState=1; m_chaosState=2;
     }

   void SetObservations(double &feat0[],double &feat1[],int n)
     {
      m_T=n;
      ArrayResize(m_obs,m_T);
      for(int t=0;t<m_T;t++){ m_obs[t][0]=feat0[t]; m_obs[t][1]=feat1[t]; }
     }

   bool IsTrained(){ return m_trained; }

   void Train(int iterations=30)
     {
      if(m_T<10) return;

      double alpha[][HMM_NSTATES];
      double beta[][HMM_NSTATES];
      double gamma[][HMM_NSTATES];
      double c[];
      ArrayResize(alpha,m_T); ArrayResize(beta,m_T); ArrayResize(gamma,m_T); ArrayResize(c,m_T);

      for(int it=0; it<iterations; it++)
        {
         double s=0;
         for(int st=0; st<HMM_NSTATES; st++){ alpha[0][st]=m_pi[st]*GaussianPDF(st,m_obs[0][0],m_obs[0][1]); s+=alpha[0][st]; }
         c[0]=(s>1e-300)?1.0/s:1.0;
         for(int st=0; st<HMM_NSTATES; st++) alpha[0][st]*=c[0];

         for(int t=1; t<m_T; t++)
           {
            s=0;
            for(int st=0; st<HMM_NSTATES; st++)
              {
               double sum=0;
               for(int pr=0; pr<HMM_NSTATES; pr++) sum+=alpha[t-1][pr]*m_A[pr][st];
               alpha[t][st]=sum*GaussianPDF(st,m_obs[t][0],m_obs[t][1]);
               s+=alpha[t][st];
              }
            c[t]=(s>1e-300)?1.0/s:1.0;
            for(int st=0; st<HMM_NSTATES; st++) alpha[t][st]*=c[t];
           }

         for(int st=0; st<HMM_NSTATES; st++) beta[m_T-1][st]=c[m_T-1];
         for(int t=m_T-2; t>=0; t--)
            for(int st=0; st<HMM_NSTATES; st++)
              {
               double sum=0;
               for(int nx=0; nx<HMM_NSTATES; nx++) sum+=m_A[st][nx]*GaussianPDF(nx,m_obs[t+1][0],m_obs[t+1][1])*beta[t+1][nx];
               beta[t][st]=sum*c[t];
              }

         for(int t=0; t<m_T; t++)
           {
            double sum=0;
            for(int st=0; st<HMM_NSTATES; st++){ gamma[t][st]=alpha[t][st]*beta[t][st]; sum+=gamma[t][st]; }
            if(sum>1e-300) for(int st=0; st<HMM_NSTATES; st++) gamma[t][st]/=sum;
           }

         for(int st=0; st<HMM_NSTATES; st++) m_pi[st]=gamma[0][st];

         double numA[HMM_NSTATES][HMM_NSTATES];
         double denA[HMM_NSTATES];
         ArrayInitialize(numA,0.0);
         ArrayInitialize(denA,0.0);
         for(int t=0; t<m_T-1; t++)
           {
            double xiT[HMM_NSTATES][HMM_NSTATES];
            double denomXi=0;
            for(int i=0;i<HMM_NSTATES;i++)
               for(int j=0;j<HMM_NSTATES;j++)
                 {
                  xiT[i][j]=alpha[t][i]*m_A[i][j]*GaussianPDF(j,m_obs[t+1][0],m_obs[t+1][1])*beta[t+1][j];
                  denomXi+=xiT[i][j];
                 }
            if(denomXi>1e-300)
               for(int i=0;i<HMM_NSTATES;i++)
                  for(int j=0;j<HMM_NSTATES;j++)
                     numA[i][j]+=xiT[i][j]/denomXi;
            for(int i=0;i<HMM_NSTATES;i++) denA[i]+=gamma[t][i];
           }
         for(int i=0;i<HMM_NSTATES;i++)
            for(int j=0;j<HMM_NSTATES;j++)
               m_A[i][j]=(denA[i]>1e-8)?numA[i][j]/denA[i]:m_A[i][j];

         for(int st=0; st<HMM_NSTATES; st++)
           {
            double denom=0,sum0=0,sum1=0;
            for(int t=0;t<m_T;t++){ denom+=gamma[t][st]; sum0+=gamma[t][st]*m_obs[t][0]; sum1+=gamma[t][st]*m_obs[t][1]; }
            if(denom>1e-8){ m_mean[st][0]=sum0/denom; m_mean[st][1]=sum1/denom; }
            double v0=0,v1=0;
            for(int t=0;t<m_T;t++)
              {
               v0+=gamma[t][st]*MathPow(m_obs[t][0]-m_mean[st][0],2);
               v1+=gamma[t][st]*MathPow(m_obs[t][1]-m_mean[st][1],2);
              }
            if(denom>1e-8){ m_var[st][0]=MathMax(v0/denom,1e-6); m_var[st][1]=MathMax(v1/denom,1e-6); }
           }
        }

      LabelStates();
      m_trained=true;
     }

   int GetCurrentStateViterbi()
     {
      if(m_T<1) return -1;
      double delta[][HMM_NSTATES];
      int    psi[][HMM_NSTATES];
      ArrayResize(delta,m_T); ArrayResize(psi,m_T);

      for(int st=0; st<HMM_NSTATES; st++)
         delta[0][st]=MathLog(MathMax(m_pi[st],1e-300))+MathLog(GaussianPDF(st,m_obs[0][0],m_obs[0][1]));

      for(int t=1; t<m_T; t++)
         for(int st=0; st<HMM_NSTATES; st++)
           {
            double best=-1e300; int bestPrev=0;
            for(int pr=0; pr<HMM_NSTATES; pr++)
              {
               double val=delta[t-1][pr]+MathLog(MathMax(m_A[pr][st],1e-300));
               if(val>best){ best=val; bestPrev=pr; }
              }
            delta[t][st]=best+MathLog(GaussianPDF(st,m_obs[t][0],m_obs[t][1]));
            psi[t][st]=bestPrev;
           }

      int lastState=0; double best=-1e300;
      for(int st=0; st<HMM_NSTATES; st++) if(delta[m_T-1][st]>best){ best=delta[m_T-1][st]; lastState=st; }
      return lastState;
     }

   string GetRegimeName(int state)
     {
      if(state==m_trendState) return "TREND";
      if(state==m_rangeState) return "RANGE";
      if(state==m_chaosState) return "CHAOTIC";
      return "UNKNOWN";
     }
  };

//--- HMM uchun kuzatuv qatorini H1 barlaridan (ADX, normallashtirilgan ATR) yig'ish
int AZZO_BuildRegimeSeries(string symbol,int barsBack,double &feat0[],double &feat1[])
  {
   int adxHandle=iADX(symbol,PERIOD_H1,14);
   int atrHandle=iATR(symbol,PERIOD_H1,14);
   if(adxHandle==INVALID_HANDLE || atrHandle==INVALID_HANDLE) return 0;

   double adxBuf[]; double atrBuf[]; double closeBuf[];
   ArraySetAsSeries(adxBuf,true);
   ArraySetAsSeries(atrBuf,true);
   ArraySetAsSeries(closeBuf,true);

   int got1=CopyBuffer(adxHandle,0,0,barsBack,adxBuf);
   int got2=CopyBuffer(atrHandle,0,0,barsBack,atrBuf);
   int got3=CopyClose(symbol,PERIOD_H1,0,barsBack,closeBuf);

   IndicatorRelease(adxHandle);
   IndicatorRelease(atrHandle);
   if(got1<=0 || got2<=0 || got3<=0) return 0;

   int n=MathMin(got1,MathMin(got2,got3));
   ArrayResize(feat0,n); ArrayResize(feat1,n);
   for(int i=0;i<n;i++)
     {
      feat0[n-1-i]=adxBuf[i];
      feat1[n-1-i]=(closeBuf[i]>0)?(atrBuf[i]/closeBuf[i])*1000.0:0.0;
     }
   return n;
  }

//--- HMM holatini UPTREND / DOWNTREND / RANGE matniga aylantiradi.
//--- TREND holatida yo'nalish Close[0] va Close[maLookback] solishtirilib aniqlanadi.
//--- RANGE va CHAOTIC ikkalasi ham vizual jihatdan "RANGE" sifatida ko'rsatiladi.
string AZZO_ClassifyDirection(CHMMRegime &hmm,string symbol,ENUM_TIMEFRAMES tf=PERIOD_H1,int maLookback=20)
  {
   int    state  = hmm.GetCurrentStateViterbi();
   string regime = hmm.GetRegimeName(state);

   if(regime=="TREND")
     {
      double closeNow  = iClose(symbol,tf,0);
      double closePast = iClose(symbol,tf,maLookback);
      if(closeNow<=0 || closePast<=0) return "RANGE";
      return (closeNow>closePast) ? "UPTREND" : "DOWNTREND";
     }
   return "RANGE";
  }

//--- YO'NALISH matniga mos rang: UPTREND=yashil, DOWNTREND=qizil, RANGE=sariq
color AZZO_DirectionColor(string dirText)
  {
   if(dirText=="UPTREND")   return C'0,255,102';   // yashil
   if(dirText=="DOWNTREND") return clrRed;         // qizil
   return clrYellow;                               // sariq (RANGE / CHAOTIC)
  }

//--- Global HMM obyekti va joriy YO'NALISH holati (panel shundan o'qiydi)
CHMMRegime g_hmm;
string     g_dirText  = "RANGE";
color      g_dirColor = clrYellow;
ulong      g_regimeLastRefreshMs = 0;
#define    REGIME_REFRESH_MS 3000   // 3 soniyada bir marta qayta hisoblanadi (har tikda emas)

//--- HMM'ni tarixiy H1 barlaridan o'qitadi (faqat OnInit'da bir marta chaqiriladi)
void AZZO_TrainRegimeModel()
  {
   double f0[], f1[];
   int n = AZZO_BuildRegimeSeries(_Symbol, 1000, f0, f1);
   if(n>10)
     {
      g_hmm.SetObservations(f0, f1, n);
      g_hmm.Train(30);
     }
  }

//--- Har REGIME_REFRESH_MS millisekundda joriy YO'NALISH'ni yangilaydi (og'ir emas)
void AZZO_UpdateRegimeIfDue()
  {
   if(!g_hmm.IsTrained()) return;
   ulong now = GetTickCount64();
   if(now - g_regimeLastRefreshMs < REGIME_REFRESH_MS) return;
   g_regimeLastRefreshMs = now;

   double f0[], f1[];
   int n = AZZO_BuildRegimeSeries(_Symbol, 300, f0, f1);
   if(n<=0) return;
   g_hmm.SetObservations(f0, f1, n);
   g_dirText  = AZZO_ClassifyDirection(g_hmm, _Symbol, PERIOD_H1, 20);
   g_dirColor = AZZO_DirectionColor(g_dirText);
  }
//+------------------------------------------------------------------+
//| AZZO_CalcMaxLot()                                                  |
//| Joriy bo'sh margin (ACCOUNT_MARGIN_FREE), leverage va instrument-  |
//| ning JORIY narxiga qarab, xavfsiz ochish mumkin bo'lgan MAKSIMAL   |
//| lot hajmini hisoblaydi. OrderCalcMargin() ishlatilgani uchun bu    |
//| FAQAT tillaga (XAUUSD) emas - HAR QANDAY brokerdagi HAR QANDAY     |
//| instrument (forex, metall, indeks, kripto) uchun to'g'ri ishlaydi, |
//| chunki kontrakt hajmi/margin valyutasi konvertatsiyasini terminal  |
//| o'zi hisoblab beradi.                                              |
//|                                                                     |
//| Masalan: XAUUSD = 4070$, leverage 1:1000 bo'lsa - 1 lot uchun      |
//| kerakli margin ~407$, ya'ni 0.01 lot uchun ~4.07$ kerak bo'ladi.   |
//| Bo'sh margin 41$ bo'lsa -> 41/4.07 =~ 10 ta 0.01 qadam -> 0.10 lot.|
//|                                                                     |
//| MUHIM: bu FAQAT panelga "MAKSIMAL LOT:" ko'rsatish uchun - robot   |
//| savdo logikasiga (NFP straddle, grid va h.k.) hech qanday ta'sir   |
//| qilmaydi, lot hajmini o'zgartirib qo'ymaydi.                       |
//+------------------------------------------------------------------+
double AZZO_CalcMaxLot()
  {
   static datetime lastLog = 0;
   bool doLog = (TimeCurrent() - lastLog >= 5); // har 5 sekundda 1 marta log

   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   if(freeMargin<=0)
     {
      if(doLog){ Print("AZZO_CalcMaxLot: freeMargin<=0 -> ", freeMargin); lastLog=TimeCurrent(); }
      return 0.0;
     }

   double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(price<=0) price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(price<=0)
     {
      if(doLog){ Print("AZZO_CalcMaxLot: price<=0 (Ask/Bid yo'q) symbol=", _Symbol); lastLog=TimeCurrent(); }
      return 0.0;
     }

   double marginPerLot = 0.0;
   bool okCalc = OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, 1.0, price, marginPerLot);

   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lotStep<=0) lotStep = 0.01;
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(minLot<=0) minLot = 0.01;
   double maxLotBroker = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(!okCalc)
     {
      if(doLog){ Print("AZZO_CalcMaxLot: OrderCalcMargin returned FALSE. GetLastError=", GetLastError(), " symbol=", _Symbol); lastLog=TimeCurrent(); }
      return 0.0;
     }

   double maxLot;
   if(marginPerLot<=0.0)
     {
      // EKSTREMAL YUQORI KREDIT HOLATI (masalan 1:2000000000):
      // 1 lot uchun margin shunchalik kichikki, terminal uni 0.0 ga
      // yumaloqlaydi. Bu "margin yo'q" degani emas - aksincha, deyarli
      // cheksiz margin degani. Shuning uchun bunda 0.00 emas, balki
      // BROKER ruxsat bergan maksimal hajmni ko'rsatamiz.
      if(doLog){ Print("AZZO_CalcMaxLot: marginPerLot=0 (ekstremal kredit) -> broker maksimal hajmi ko'rsatiladi: ", maxLotBroker); lastLog=TimeCurrent(); }
      maxLot = (maxLotBroker>0) ? maxLotBroker : 0.0;
     }
   else
     {
      double rawLots   = freeMargin / marginPerLot;
      double stepsFloor = MathFloor(rawLots/lotStep + 1e-8);
      maxLot     = stepsFloor*lotStep;

      if(maxLot < minLot)
        {
         if(doLog){ Print("AZZO_CalcMaxLot: maxLot(",maxLot,") < minLot(",minLot,") -> 0.0. freeMargin=",freeMargin," marginPerLot=",marginPerLot," rawLots=",rawLots); lastLog=TimeCurrent(); }
         maxLot = 0.0;                                // minimal lotni ham ocholmaydi
        }
     }

   if(maxLotBroker>0 && maxLot>maxLotBroker) maxLot = maxLotBroker;  // broker chegarasidan oshmasin

   int lotDigits = (lotStep<0.01) ? 3 : 2;
   return NormalizeDouble(maxLot, lotDigits);
  }
//+------------------------------------------------------------------+
//| AZZO ML MODULE TUGADI                                             |
//+------------------------------------------------------------------+

//--- Asosiy "AZZO NEON PRO" panelining o'lchamlari (bitta joyda saqlanadi, shunda
//--- MA'LUMOT (INFO) paneli ham xuddi shu balandlikda "parallel" chizilishi kafolatlanadi)
#define MAIN_PANEL_WIDTH   260
#define MAIN_PANEL_HEIGHT  502 // 426 -> 452 -> 447 -> 481 -> 502: "SCREENSHOOT: TRUE/FALSE" tugmasi qo'shildi (MAKS LOT bilan ORDER QO'YISH orasida)

enum ENUM_PANEL_CORNER
  {
   Corner_Left_Upper   = CORNER_LEFT_UPPER,   
   Corner_Right_Upper  = CORNER_RIGHT_UPPER,  
   Corner_Left_Lower   = CORNER_LEFT_LOWER,   
   Corner_Right_Lower  = CORNER_RIGHT_LOWER   
  };

//+------------------------------------------------------------------+
//| RECOVERY GRID uchun ishlash rejimi (2-BOSQICH / HOLAT sozlamasi)   |
//|   InpRecoveryBlockStopOrders = true  -> Recovery Grid ISHLAYDI,    |
//|                                  Stop VA Limit orderlar qo'yiladi  |
//|   InpRecoveryBlockStopOrders = false (DEFAULT) -> HECH QANDAY      |
//|                                  recovery order qo'yilmaydi        |
//+------------------------------------------------------------------+

//--- ROBOT SOZLAMALARI
input group "--- Azzo Trade: NFP_KILLER_2607 ---"
input bool          InpTradeNFP          = true;          
input string        InpNewsTime          = "15:29:55";    
input int           InpTimeWindowSeconds = 30;   // Maqsadli vaqtdan keyin ORDER OCHISH uchun ruxsat etilgan oyna (soniyada). 30 = faqat 30 soniya ichida.
input int           InpMaxOrders         = 15;              

input group "--- Grid Orderlari Orasidagi Masofa (1-5 va 6-dan keyingi alohida) ---"
input int           InpGridStepGroup1        = 70;   // 1-5 orderlar O'ZARO orasidagi masofa (PUNKT)
input int           InpGridStepGroup2        = 70;   // 6-dan keyingi orderlar O'ZARO orasidagi masofa (PUNKT)
input int           InpGridGapBetweenGroups  = 330;  // 5-order bilan 6-order ORASIDAGI (guruhlar orasidagi) QO'SHIMCHA masofa (PUNKT) = 3.3$

input group "--- Har bir Grid Darajasi Lot Sozlamalari (1-10) ---"
input double        InpLotLevel1         = 0.02;           
input double        InpLotLevel2         = 0.02;           
input double        InpLotLevel3         = 0.02;           
input double        InpLotLevel4         = 0.02;           
input double        InpLotLevel5         = 0.02;           
input double        InpLotLevel6         = 0.01;           
input double        InpLotLevel7         = 0.01;           
input double        InpLotLevel8         = 0.01;           
input double        InpLotLevel9         = 0.01;           
input double        InpLotLevel10        = 0.01;           

input group "--- Order Masofasi va Cheklovlar ---"
input int           InpDistance          = 750;            
input int           InpStopLoss          = 750;            
input int           InpTakeProfit        = 9000;            

input group "--- Pending Order Zones (Vizual Zonalar - BUY/SELL STOP joylashuvi) ---"
input bool          InpZonesShowOnStart  = false;         // EA ishga tushganda zonalar avtomatik ko'rsatilsinmi
input int           InpZoneOpacity       = 60;             // "Shaffoflik" %% (0-100) - Entry (sariq) chiziqlar uchun. Kichik = xiralashgan/to'q, katta = yorqin
input int           InpZoneWidth         = 3;               // Zona chiziqlari qalinligi (px)
input int           InpZoneCandlesSide   = 12;              // Narxdan chapga VA o'ngga nechta sham (candle) uzunlikda chiziq chizilsin (har yangi sham ochilganda o'zi shu songa moslab siljiydi)
input int           InpSpreadOpacity     = 50;               // Bid va Ask oralig'idagi KULRANG zonaning "shaffoflik" %% i (0-100)

input group "--- Himoya va Slippage Sozlamalari ---"
input int           InpMaxSlippage       = 3000;          
input bool          InpUseOCO            = true;            
input bool          InpUseTrailing       = false;          
input int           InpTrailingStart     = 400;            
input int           InpTrailingStep      = 200;            
input ulong         MAGIC_NUMBER         = 777777;         

input group "--- Recovery / Averaging Grid (Foydadan keyin qo'shimcha pending orderlar) ---"
input bool          InpRecoveryEnable       = true;          // Yoqish/o'chirish (har doim yoniq turadi)
input int           InpRecoveryTriggerPoints= 200;           // Trigger: entry narxidan shuncha PUNKT foyda tomonga siljisa faollashadi (100 punkt = $1, ya'ni 200 = $2)
input int           InpRecoveryOffsetPoints = 100;           // Birinchi daraja joriy narxdan qancha PUNKT uzoqlikda (100 = $1)
input int           InpRecoveryStepPoints   = 70;             // Har bir daraja orasidagi masofa PUNKTDA (70 = $0.70)
input int           InpRecoveryLevels       = 10;             // Har tomonda (yuqori/past) nechta pending order
input double        InpRecoveryLot          = 0.01;           // Har bir daraja loti

input group "--- Recovery Grid, 2-BOSQICH: Bloklash rejimi ---"
input bool          InpRecoveryBlockStopOrders = false; // HOLAT (yoqish/o'chirish): TRUE - Recovery Grid ishlaydi (Stop+Limit qo'yiladi), FALSE - hech qanday order qo'yilmaydi

input group "--- PANEL JOYLASHUVI VA HOTKEYS ---"
input ENUM_PANEL_CORNER InpPanelCorner    = Corner_Left_Lower; 
input int                InpOffsetX        = 15;                
input int                InpOffsetY        = 15;                
input bool               InpUseHotkeys     = true;              
input bool               InpHideChartHistory = true;   // Grafikda savdo tarixi (qizil/ko'k strelkalar) YASHIRILSINMI?

input group "--- Telegram Bildirishnomalari ---"
input bool          InpTelegramEnabled    = false;        // Yoqish/o'chirish (token qo'yilmaguncha OFF qoldiring)
input string        InpTelegramBotToken   = "8914631795:AAFqhLZVqSEgx7diDjLHXFQh_qwdhrqCT1o"; // BotFather'dan olingan token
input string        InpTelegramChatID     = "7030423994"; // Sizning yoki guruhingiz Chat ID raqami
input bool          InpTelegramOnOpen     = true;          // Order OCHILGANDA xabar yuborish
input bool          InpTelegramOnClose    = true;          // Order YOPILGANDA xabar yuborish
input bool          InpTelegramOnStraddle = true;          // NFP straddle qo'yilganda xabar yuborish

input group "--- NFP OLDIN OGOHLANTIRISH (Pre-Alert) ---"
input bool          InpPreAlertEnabled  = true;    // Yoqish/o'chirish
input int           InpPreAlertMinutes  = 5;       // NFP dan necha DAQIQA oldin ogohlantirilsin
input string        InpPreAlertMessage  = "⚠️ DIQQAT! NFP 5 daqiqadan keyin e'lon qilinadi! Tayyor turing!"; // MT5 Alert va Telegramga yuboriladigan matn (o'zingiz xohlagancha o'zgartiring)

input group "--- Yangiliklar Kalendari (F paneli) - MT5 ICHKI ECONOMIC CALENDAR ---"
input string        InpNewsCurrencies       = "NZD,USD"; // Qaysi valyutalar kerak (vergul bilan ajratilgan, masalan: USD,EUR,GBP). Bo'sh qoldirilsa - HAMMASI ko'rsatiladi
input int           InpNewsCalendarCount    = 6;     // Nechta TOP (qizil/muhim) yangilik ko'rsatilsin (1-10)
input int           InpNewsLookAheadDays    = 14;     // Necha kun oldinga qarab yangilik qidirilsin (faqat filtr uchun)
input int           InpNewsRefreshSeconds   = 2;      // Nechchi soniyada bir marta MT5 ichki kalendaridan qayta o'qilsin (mahalliy chaqiruv - tez bo'lishi xavfsiz; countdown BUNGA BOG'LIQ EMAS - u alohida, uzluksiz real vaqtda yuriladi)

// --- RECOVERY GRID: joriy holatni kuzatish uchun global o'zgaruvchilar ---
// g_recoveryDirection: 0 = pozitsiya yo'q, 1 = BUY grid, 2 = SELL grid
int  g_recoveryDirection   = 0;
bool g_recoveryGridActive  = false;   // joriy tsikl uchun recovery orderlar allaqachon qo'yilganmi
#define AZZO_RECOVERY_MAGIC_OFFSET 1000

// --- YANGI: asosiy grid uchun "Orderlar soni" maydonidagi MAKSIMAL qiymat.
// --- Avval bu 10 bilan qattiq cheklangan edi. Endi 50 gacha ruxsat etiladi
// --- (broker/terminalning pending order limitlari va panel ko'rinishi uchun
// --- xavfsiz, lekin AMALDA "cheksizga yaqin" son). Xohlasangiz shu qiymatni
// --- pastda o'zgartirib, yanada oshirishingiz mumkin - lekin MAGIC_NUMBER +
// --- AZZO_RECOVERY_MAGIC_OFFSET (1000) dan oshib ketmasligi kerak, aks holda
// --- asosiy grid va recovery-grid orderlari MAGIC raqami ustma-ust tushib qoladi.
#define AZZO_MAX_GRID_ORDERS 50
// INFO paneldagi qatorlar soni uchun xavfsiz yuqori chegara (1-5 alohida +
// 5 dan keyingi orderlar uchun BITTA yagona guruh qatori = eng ko'pi 6 qator).
#define AZZO_MAX_INFO_ROWS 6

// Global o'zgaruvchilar
bool ordersPlaced = false;
bool globalTradeNFP = true;
bool panelMinimized = false; 
bool infoPanelOpen = false;                                
bool countdownPanelOpen = false;   // "TESKARI SANOQ" (Countdown) paneli - "T" tugmasi bilan ochiladi/yopiladi
bool newsPanelOpen = false;        // "YANGILIKLAR KALENDARI" paneli - "F" tugmasi bilan ochiladi/yopiladi
bool retracementPanelOpen = false; // "RETRACEMENT" (LIMIT ORDERS / Recovery Grid) paneli - "R" tugmasi bilan ochiladi/yopiladi
bool g_zonesVisible = false;       // Pending Order Zones (Buy/Sell Stop, SL, TP chiziqlari) - "P" tugmasi bilan ON/OFF
string prefix = "AzzoNeon_";

int last_chart_w = -1;
int last_chart_h = -1;
ENUM_PANEL_CORNER currentCorner = Corner_Left_Lower;
ENUM_PANEL_CORNER last_corner = (ENUM_PANEL_CORNER)-1;
double inputLots[AZZO_MAX_GRID_ORDERS];                    

// --- INFO panelidagi "Orderlar soni" maydoni orqali RUNTIME'DA (dastur ishlab turganda)
// --- o'zgartiriladigan order-soni. 1 dan AZZO_MAX_GRID_ORDERS (50) gacha bo'lishi
// --- mumkin. OnInit()'da InpMaxOrders qiymatidan boshlang'ich holatga keltiriladi,
// --- keyin esa faqat panel orqali (info_ordercount edit maydoni) o'zgartiriladi.
int g_orderCount = 5;

// --- RETRACEMENT (LIMIT ORDERS) panelidagi tahrirlanadigan Recovery Grid sozlamalari.
// --- OnInit() da tegishli Inp* qiymatlaridan boshlang'ich holatga keltiriladi,
// --- keyin esa FAQAT "RETRACEMENT [R]" paneli orqali (qayta compile qilmasdan)
// --- o'zgartiriladi - xuddi g_orderCount / g_nfpTime kabi.
int    g_recTrigger = 200;   // Trigger: entry narxidan shuncha PUNKT foyda tomonga siljisa faollashadi
int    g_recOffset  = 100;   // Birinchi daraja joriy narxdan qancha PUNKT uzoqlikda
int    g_recStep    = 70;    // Har bir daraja orasidagi masofa PUNKTDA
int    g_recLevels  = 10;    // Har tomonda (yuqori/past) nechta pending order
double g_recLot      = 0.01; // Har bir daraja loti

// --- YANGI: INFO panelidagi "orderlar orasidagi masofa" (Grid Step) tahrirlanadigan
// --- maydonlari. OnInit()da InpGridStepGroup1 / InpGridStepGroup2 / InpGridGapBetweenGroups
// --- qiymatlaridan boshlang'ich holatga keltiriladi, keyin esa FAQAT INFO paneli orqali
// --- (qayta compile qilmasdan) o'zgartiriladi - xuddi g_orderCount / g_recStep kabi.
int g_gridStep1 = 70;   // 1-5 orderlar O'ZARO orasidagi masofa (PUNKT)
int g_gridStep2 = 70;   // 6-50 orderlar O'ZARO orasidagi masofa (PUNKT)
int g_gridGap   = 250;  // 5-order bilan 6-order ORASIDAGI (guruhlar orasidagi) QO'SHIMCHA masofa (PUNKT)

// --- YANGI: "MASOFASI" (InpDistance) va "SL" (InpStopLoss) endi INFO panelidagi
// --- "ORDERLAR ORASI" blokining PASTIDA, RUNTIME'DA (qayta compile qilmasdan)
// --- tahrirlanadi - xuddi g_gridStep1/g_gridStep2/g_gridGap kabi. OnInit()da
// --- Inp* qiymatlaridan (ikkalasi ham 750 = 7.5$) boshlang'ich holatga
// --- keltiriladi, keyin esa FAQAT panel maydoni orqali o'zgaradi.
int g_orderDistance = 750; // Order narxining joriy Ask/Bid'dan uzoqligi (PUNKT)
int g_orderSL       = 750; // Stop Loss masofasi (PUNKT)

// --- YANGI: "HOLAT" (TRUE/FALSE) tugmasi orqali RUNTIME'DA (qayta compile
// --- qilmasdan) yoqiladi/o'chiriladi. OnInit()da InpRecoveryBlockStopOrders
// --- qiymatidan boshlang'ich holatga keltiriladi, keyin esa FAQAT "HOLAT"
// --- tugmasini bosish orqali TRUE <-> FALSE almashadi.
bool g_recBlockStopOrders = false;

// --- YANGILIKLAR KALENDARI (F paneli) uchun ma'lumotlar. RefreshNewsCalendar()
// --- endi tashqi WebRequest'ga UMUMAN bog'liq emas - MT5 terminalining O'ZI
// --- fon rejimida avtomatik sinxronlaydigan ICHKI Economic Calendar (Calendar*
// --- funksiyalari: CalendarValueHistory/CalendarEventById/CalendarCountryById)
// --- dan to'g'ridan-to'g'ri o'qiydi. Bu 100% MAHALLIY (lokal) chaqiruv bo'lgani
// --- uchun tarmoq so'rovi umuman yuborilmaydi - shuning uchun HECH QACHON
// --- qotmaydi va URL'ni WebRequest ruxsat ro'yxatiga qo'shish shart emas.
#define NEWS_MAX_ROWS 10
datetime g_newsDateTime[NEWS_MAX_ROWS];
string   g_newsDay[NEWS_MAX_ROWS];
string   g_newsDate[NEWS_MAX_ROWS];
string   g_newsTime[NEWS_MAX_ROWS];
string   g_newsCurrency[NEWS_MAX_ROWS];
string   g_newsName[NEWS_MAX_ROWS];
int      g_newsCount = 0;
bool     g_newsDataReady = false;   // kamida bitta muvaffaqiyatli yangilash bo'lganmi
ulong    g_newsLastRefreshMs = 0;

void   RefreshNewsCalendar();
void   UpdateNewsPanelTexts();
void   UpdateNewsCountdownLine();
string DayNameFF(datetime t);
bool   IsNewsCurrencyWanted(string curCode);

//+------------------------------------------------------------------+
//| TELEGRAM - "BATCH" (to'plam) tizimi                                |
//| Bir necha order bir zumda (masalan bir necha yuz millisekund      |
//| ichida) ochilsa yoki yopilsa, ularni ALOHIDA emas, BITTA umumiy   |
//| xabarda yuborish uchun ishlatiladi. Har bir yangi hodisa "vaqt     |
//| hisoblagichini" qayta boshlaydi; BATCH_FLUSH_MS ichida yangi       |
//| hodisa kelmasa, to'plangan hammasi bitta xabar qilib yuboriladi.  |
//+------------------------------------------------------------------+
#define BATCH_FLUSH_MS 700

string  g_openBatchType[];
double  g_openBatchLot[];
double  g_openBatchPrice[];
double  g_openBatchProfit[];
bool    g_openBatchActive  = false;
ulong   g_openBatchLastMs  = 0;

string  g_closeBatchType[];
double  g_closeBatchLot[];
double  g_closeBatchOpenPrice[];
double  g_closeBatchClosePrice[];
double  g_closeBatchProfit[];
int     g_closeBatchPoints[];
bool    g_closeBatchActive = false;
ulong   g_closeBatchLastMs = 0;

void FlushOpenBatch();
void FlushCloseBatch();

// --- NFP VAQTINI PANEL ORQALI (dastur ishlab turganda) o'zgartirish uchun ---
// g_nfpTime - haqiqatda ishlatiladigan NFP vaqti. InpNewsTime faqat BOSHLANG'ICH
// (dastlabki) qiymat sifatida OnInit()'da o'qiladi, keyin esa faqat panel
// ustidagi "val_trig" edit maydoni orqali (qayta compile qilmasdan) o'zgartiriladi.
string g_nfpTime = "15:29:55";
datetime g_lastPreAlertTargetTime = 0; // Har bir NFP maqsad vaqti uchun ogohlantirish FAQAT BIR MARTA yuborilishini kafolatlaydi

int robotLang = 1;      // 1 = UZB, 2 = ENG, 3 = RUS
int currentTheme = 4;   
bool rgbMode = false;   

// --- PANELNI SICHQONCHA BILAN SUDRASH (DRAG & DROP) UCHUN O'ZGARUVCHILAR ---
// g_manualPos = true bo'lsa, panel endi InpPanelCorner burchagiga emas,
// balki foydalanuvchi qo'yib ketgan g_panelX/g_panelY absolyut piksel
// koordinatasiga joylashadi. Ctrl+Strelka bosilsa, avtomatik burchakka
// qaytariladi (g_manualPos qayta false qilinadi).
bool g_manualPos       = false;
int  g_panelX          = 0;
int  g_panelY          = 0;
bool g_dragActive      = false;
int  g_dragStartMouseX = 0;
int  g_dragStartMouseY = 0;
int  g_dragStartPanelX = 0;
int  g_dragStartPanelY = 0;
// --- YANGI (performance fix): sudrash paytida RelocatePanel()+ChartRedraw()
// --- har MOUSE_MOVE hodisasida emas, balki max ~60 fps (16ms) chastotada
// --- ishlashi uchun "throttle" vaqt belgisi. Buning sababi: MetaTrader
// --- sichqoncha harakatini juda tez-tez yuboradi; agar har bir hodisada
// --- butun panel (30-90 obyekt) qayta joylashtirilsa, hodisalar navbatga
// --- to'planib qoladi va panel sichqonchadan "orqada qolib", keyin sakrab
// --- "quvib yetadi" - foydalanuvchi buni "0.5 sekund kechikish/qotish"
// --- sifatida ko'radi.
uint g_lastDragRedrawMs = 0;

// --- VIZUAL TUGMA HOVER FX ------------------------------------------------
// Sichqoncha OBJ_BUTTON tugmalardan biri ustiga kelganda fon/border rangi
// silliq (bir necha OnTimer tiki davomida, 50ms tikda ~step ulushi) bazaviy
// rangdan neon "glow" rangiga o'tadi; sichqoncha chiqib ketganda xuddi
// shunday silliq orqaga qaytadi. Progress 0.0 = odatiy holat, 1.0 = to'liq
// hover (neon) holat. ColorBlend() (pastda mavjud) orqali interpolyatsiya
// qilinadi.
#define HOVER_BTN_COUNT 9   // 6 -> 9: "NFP"/"CPI"/"OTHER NEWS" preset tugmalari uchun hover FX qo'shildi
string g_hoverBtnName[HOVER_BTN_COUNT];
color  g_hoverBaseBg[HOVER_BTN_COUNT];
color  g_hoverBaseBorder[HOVER_BTN_COUNT];
color  g_hoverGlowBg[HOVER_BTN_COUNT];
color  g_hoverGlowBorder[HOVER_BTN_COUNT];
double g_hoverProgress[HOVER_BTN_COUNT];
bool   g_hoverTarget[HOVER_BTN_COUNT];
bool   g_hoverBtnsRegistered = false;

double fadeProgress = 0.0;
int currentTargetTheme = 5;
color ColorMainNeon;
color ColorBorderNeon;
color ColorHeaderBg;
color ColorFooterBg;
color ColorMainBg = C'12,12,12';

color ColorTextPrimary = clrWhite;    // panel ichidagi oddiy (asosiy) matn rangi
color ColorChipBg      = C'22,22,22'; // minimallashtirish/tema kabi kichik "chip" tugmalar foni

// --- FOREX FACTORY (F) paneli uchun DOIMIY sayt-uslubidagi ranglar. Foydalanuvchi
// --- tanlagan NEON mavzudan (ApplyTheme) MUSTAQIL - F paneli har doim ForexFactory.com
// --- saytiga o'xshab to'q ko'k (navy) sarlavha + qizil "muhim yangilik" belgisi bilan
// --- chiziladi, teма o'zgarganda ham ko'rinishi o'zgarmaydi.
#define FF_BG       C'12,12,12'     // panel foni - ASOSIY panel (ColorMainBg) bilan AYNAN bir xil
#define FF_HEADER   C'22,22,22'     // sarlavha foni - ASOSIY panel (ColorHeaderBg) bilan AYNAN bir xil
#define FF_BORDER   C'40,64,97'     // (endi ishlatilmaydi - chegara rangi ColorMainNeon'dan olinadi)
#define FF_RED      C'214,38,38'    // "High Impact" qizil rangi (muhim yangilik urg'usi)
#define FF_CYAN     C'0,220,255'    // sana/kun/vaqt qatori uchun DEFAULT rang (qizil emas, cyan)
#define FF_STRIPE_A C'18,18,18'     // juft qatorlar foni (zebra) - asosiy panel kulrang gammasida
#define FF_STRIPE_B C'12,12,12'     // toq qatorlar foni (zebra) - asosiy panel kulrang gammasida
#define FF_CURBG    C'30,30,30'     // valyuta "badge"i foni - asosiy panel kulrang gammasida
#define FF_SUBTEXT  C'150,150,150'  // sana/vaqt kabi ikkinchi darajali matn rangi (neytral kulrang)

uchar ThemeRGB[10][3] = {
   {0, 0, 0},         
   {210, 215, 220},   
   {0, 255, 255},     
   {57, 255, 20},     
   {0, 204, 255},     
   {255, 255, 0},     
   {255, 0, 128},     
   {157, 0, 255},     
   {255, 102, 0},     
   {255, 0, 55}       
};

// Funksiyalar prototiplari
void ApplyTheme(int themeNum);
void ApplyPanelBaseColors();
void ApplySmoothRGB();
double NormalizeLot(double lot);
int ParseNfpTimeInput(string text, string &outNormalized);
int BuildInfoRows(int orderCount, int &rowStart[], int &rowEnd[]);
double GetGridOffsetPoints(int orderIndex0based);
void CreateDashboard();
void RelocatePanel(bool force = false);
void UpdateDashboard();
void UpdateCountdownPanel();
void UpdateMinimizedCountdown();
void CheckNfpPreAlert();
void PlaceStraddleOrders();
void PlaceManualOrders();
void CloseAllPositionsAndOrders();
void CheckSmartOCO(); 
void ApplyTrailingStop();
void ManageRecoveryGrid();
void PlaceRecoveryGrid(int dir, double currentProfitPoints);
void DeleteRecoveryPendingOrders();

// --- YANGI: MA'LUMOT (INFO) panelidagi tayyor sozlamalar (PRESET) tugmalari ---
// "NFP", "CPI" va "OTHER NEWS" tugmalari bosilganda barcha Order/Lot, Grid Step,
// Masofa va SL maydonlarini bir zumda oldindan belgilangan qiymatlarga o'rnatadi
// - qayta compile qilmasdan, panelning o'zida darhol.
void AZZO_ApplyPreset(int orderCount, double lotFirst5, double lotRest, int step1, int step2, int gap, int distance, int sl);
void AZZO_ApplyPresetNFP();
void AZZO_ApplyPresetCPI();
void AZZO_ApplyPresetOtherNews();

// --- Visual Button Hover FX ---
void InitHoverButtons();
void UpdateButtonHoverHitTest(int mouseX, int mouseY);
void UpdateButtonHoverFX();

// --- Pending Order Zones (vizual: Buy/Sell Stop, SL, TP chiziqlari) ---
color ColorBlend(color fg, color bg, double opacityPercent);
void  DrawZoneLine(string name, double price, color clr, int width, ENUM_LINE_STYLE style);
void  DrawSpreadZone();
void  DrawPendingZones();
void  DeleteAllZoneLines();

void CreateObjRect(string name, int w, int h, color bg_color, ENUM_BASE_CORNER corner, int zorder);
void CreateObjText(string name, string text, int size, string font, color clr, ENUM_BASE_CORNER corner, int zorder);
void CreateObjButton(string name, string text, int w, int h, color bg_color, color txt_color, ENUM_BASE_CORNER corner, int zorder);
void CreateObjEdit(string name, string text, int w, int h, color bg_color, color txt_color, ENUM_BASE_CORNER corner, int zorder);
void ObjSetXY(string name, int x, int y);

//+------------------------------------------------------------------+
//| TELEGRAM BILDIRISHNOMALARI - funksiya prototiplari                |
//+------------------------------------------------------------------+
void   SendTelegramMessage(string text);
string UrlEncode(string text);
string FormatUzTime(datetime t);

//+------------------------------------------------------------------+
//| Yopilgan pozitsiyaning ochilish narxini topish uchun (Telegram    |
//| "order yopildi" xabarida Pts hisoblash uchun ishlatiladi).        |
//+------------------------------------------------------------------+
double FindPositionOpenPrice(ulong positionId);

//+------------------------------------------------------------------+
//| ApplyPanelBaseColors()                                             |
//| Panelning ASOSIY fon/sarlavha/matn ranglarini doim QORA (dark)     |
//| interfeysga belgilaydi. ColorMainNeon/ColorBorderNeon (foydalanuvchi|
//| tanlagan NEON aksent rang, 1-9 tema tugmalari) bu yerda            |
//| O'ZGARMAYDI - faqat fon va oddiy matn ranglari o'rnatiladi.        |
//+------------------------------------------------------------------+
void ApplyPanelBaseColors()
  {
   ColorMainBg      = C'12,12,12';    // qora asosiy fon
   ColorHeaderBg    = C'22,22,22';
   ColorFooterBg    = C'16,16,16';
   ColorChipBg      = C'22,22,22';
   ColorTextPrimary = clrWhite;
  }

void ApplyTheme(int themeNum)
  {
   currentTheme = themeNum;
   ApplyPanelBaseColors();

   uchar r = ThemeRGB[themeNum][0];
   uchar g = ThemeRGB[themeNum][1];
   uchar b = ThemeRGB[themeNum][2];
   
   ColorMainNeon = StringToColor(StringFormat("%d,%d,%d", r, g, b));
   ColorBorderNeon = StringToColor(StringFormat("%d,%d,%d", (int)(r*0.6), (int)(g*0.6), (int)(b*0.6)));
  }

void ApplySmoothRGB()
  {
   ApplyPanelBaseColors();
   fadeProgress += (50.0 / 1600.0);
   
   if(fadeProgress >= 1.0)
     {
      fadeProgress = 0.0;
      currentTheme = currentTargetTheme;
      currentTargetTheme = currentTheme + 1;
      if(currentTargetTheme > 9) currentTargetTheme = 1;
     }

   int r1 = ThemeRGB[currentTheme][0]; int g1 = ThemeRGB[currentTheme][1]; int b1 = ThemeRGB[currentTheme][2];
   int r2 = ThemeRGB[currentTargetTheme][0]; int g2 = ThemeRGB[currentTargetTheme][1]; int b2 = ThemeRGB[currentTargetTheme][2];

   int currR = (int)(r1 + (r2 - r1) * fadeProgress);
   int currG = (int)(g1 + (g2 - g1) * fadeProgress);
   int currB = (int)(b1 + (b2 - b1) * fadeProgress);

   ColorMainNeon = StringToColor(StringFormat("%d,%d,%d", currR, currG, currB));
   ColorBorderNeon = StringToColor(StringFormat("%d,%d,%d", (int)(currR*0.6), (int)(currG*0.6), (int)(currB*0.6)));

   ObjectSetInteger(0, prefix+"border_neon", OBJPROP_COLOR, ColorMainNeon);
   ObjectSetInteger(0, prefix+"title_text", OBJPROP_COLOR, ColorMainNeon);
   if(ObjectFind(0, prefix+"brand_txt") >= 0) ObjectSetInteger(0, prefix+"brand_txt", OBJPROP_COLOR, ColorMainNeon);
   if(ObjectFind(0, prefix+"min_border") >= 0) ObjectSetInteger(0, prefix+"min_border", OBJPROP_COLOR, ColorMainNeon);

   ObjectSetInteger(0, prefix+"btn_set", OBJPROP_COLOR, ColorMainNeon);

   if(ObjectFind(0, prefix+"info_bg") >= 0)
     {
      ObjectSetInteger(0, prefix+"info_border", OBJPROP_COLOR, ColorMainNeon);
      ObjectSetInteger(0, prefix+"info_title", OBJPROP_COLOR, ColorMainNeon);
      for(int i=0; i<AZZO_MAX_INFO_ROWS; i++)
        {
         if(ObjectFind(0, prefix+"info_lbl_"+(string)i) >= 0) ObjectSetInteger(0, prefix+"info_lbl_"+(string)i, OBJPROP_COLOR, ColorBorderNeon);
         if(ObjectFind(0, prefix+"info_val_"+(string)i) >= 0) ObjectSetInteger(0, prefix+"info_val_"+(string)i, OBJPROP_COLOR, ColorMainNeon);
         if(ObjectFind(0, prefix+"info_unit_"+(string)i) >= 0) ObjectSetInteger(0, prefix+"info_unit_"+(string)i, OBJPROP_COLOR, ColorBorderNeon);
        }
      if(ObjectFind(0, prefix+"info_oc_lbl") >= 0) ObjectSetInteger(0, prefix+"info_oc_lbl", OBJPROP_COLOR, ColorBorderNeon);
      if(ObjectFind(0, prefix+"info_ordercount") >= 0) ObjectSetInteger(0, prefix+"info_ordercount", OBJPROP_COLOR, ColorMainNeon);

      if(ObjectFind(0, prefix+"info_grid_title") >= 0) ObjectSetInteger(0, prefix+"info_grid_title", OBJPROP_COLOR, ColorMainNeon);
      for(int gi=1; gi<=3; gi++)
        {
         if(ObjectFind(0, prefix+"info_grid_lbl_"+(string)gi) >= 0)  ObjectSetInteger(0, prefix+"info_grid_lbl_"+(string)gi, OBJPROP_COLOR, ColorBorderNeon);
         if(ObjectFind(0, prefix+"info_grid_val_"+(string)gi) >= 0)  ObjectSetInteger(0, prefix+"info_grid_val_"+(string)gi, OBJPROP_COLOR, ColorMainNeon);
         if(ObjectFind(0, prefix+"info_grid_unit_"+(string)gi) >= 0) ObjectSetInteger(0, prefix+"info_grid_unit_"+(string)gi, OBJPROP_COLOR, ColorBorderNeon);
        }
     }

   // --- RETRACEMENT (LIMIT ORDERS) paneli - endi INFO panel bilan BIR XIL rangda
   // --- (label ColorBorderNeon, edit matni ColorMainNeon) - shuning uchun tema
   // --- almashganda ham INFO bilan TO'LIQ SINXRON yangilanadi. ---
   if(ObjectFind(0, prefix+"retr_bg") >= 0)
     {
      ObjectSetInteger(0, prefix+"retr_border", OBJPROP_COLOR, ColorMainNeon);
      ObjectSetInteger(0, prefix+"retr_title", OBJPROP_COLOR, ColorMainNeon);
      for(int ri=0; ri<5; ri++)
        {
         if(ObjectFind(0, prefix+"retr_lbl_"+(string)ri) >= 0) ObjectSetInteger(0, prefix+"retr_lbl_"+(string)ri, OBJPROP_COLOR, ColorBorderNeon);
         if(ObjectFind(0, prefix+"retr_val_"+(string)ri) >= 0) ObjectSetInteger(0, prefix+"retr_val_"+(string)ri, OBJPROP_COLOR, ColorMainNeon);
        }
      // --- "HOLAT" (Status) qatorining label'i boshqa label'lar bilan bir xil
      // --- temaga ergashadi, lekin QIYMATI (TRUE=yashil / FALSE=qizil) doim
      // --- o'z holatini ko'rsatishi kerak - shuning uchun tema ranggia BOG'LANMAYDI.
      if(ObjectFind(0, prefix+"retr_lbl_status") >= 0) ObjectSetInteger(0, prefix+"retr_lbl_status", OBJPROP_COLOR, ColorBorderNeon);
     }

   // --- TESKARI SANOQ (Countdown, "T") paneli - ASOSIY panel bilan bir xil ------------
   // RGB rejimida bu panel ham ochiq bo'lsa, chegara va sarlavha rangi ASOSIY paneldagi
   // ColorMainNeon bilan HAR FREYMDA birga yangilanadi (avval "qotib qolar" edi).
   if(ObjectFind(0, prefix+"cd_bg") >= 0)
     {
      if(ObjectFind(0, prefix+"cd_border") >= 0) ObjectSetInteger(0, prefix+"cd_border", OBJPROP_COLOR, ColorMainNeon);
      if(ObjectFind(0, prefix+"cd_title")  >= 0) ObjectSetInteger(0, prefix+"cd_title",  OBJPROP_COLOR, ColorMainNeon);
     }

   // --- FOREX FACTORY (F.FACTORY, "F") paneli - ASOSIY/INFO panel bilan bir xil ------
   // Fon/sarlavha foni (FF_BG, FF_HEADER) doimiy qoladi (ForexFactory uslubi), lekin
   // chegara va sarlavha MATNI rangi ASOSIY paneldagi neon rang bilan sinxron bo'lishi
   // kerak - shuning uchun bu ham har freymda ColorMainNeon'ga yangilanadi.
   if(ObjectFind(0, prefix+"news_bg") >= 0)
     {
      if(ObjectFind(0, prefix+"news_border") >= 0) ObjectSetInteger(0, prefix+"news_border", OBJPROP_COLOR, ColorMainNeon);
      if(ObjectFind(0, prefix+"news_title")  >= 0) ObjectSetInteger(0, prefix+"news_title",  OBJPROP_COLOR, ColorMainNeon);
      for(int i=0; i<NEWS_MAX_ROWS; i++)
        {
         string curBgName = prefix+"news_curbg_"+(string)i;
         if(ObjectFind(0, curBgName) >= 0) ObjectSetInteger(0, curBgName, OBJPROP_COLOR, ColorMainNeon);
        }
     }

   ObjectSetInteger(0, prefix+"lbl_sym", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_bal", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_spd", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_slp", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_lev", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_loc", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_srv", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_trig", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_win", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_ping", OBJPROP_COLOR, ColorBorderNeon); 
   ObjectSetInteger(0, prefix+"lbl_dir", OBJPROP_COLOR, ColorBorderNeon);
   ObjectSetInteger(0, prefix+"lbl_maxlot", OBJPROP_COLOR, ColorBorderNeon);

   ObjectSetInteger(0, prefix+"val_sym", OBJPROP_COLOR, ColorMainNeon);
   ObjectSetInteger(0, prefix+"val_bal", OBJPROP_COLOR, ColorMainNeon);
   ObjectSetInteger(0, prefix+"val_spd", OBJPROP_COLOR, ColorMainNeon);
   ObjectSetInteger(0, prefix+"val_slp", OBJPROP_COLOR, ColorMainNeon);
   ObjectSetInteger(0, prefix+"val_lev", OBJPROP_COLOR, ColorMainNeon);
   ObjectSetInteger(0, prefix+"val_win", OBJPROP_COLOR, ColorMainNeon);
   ObjectSetInteger(0, prefix+"val_ping", OBJPROP_COLOR, ColorMainNeon); 
  }

//+------------------------------------------------------------------+
//| FindPositionOpenPrice()                                           |
//| Looks up the entry price of the position a closing deal belongs   |
//| to, so the history table can show real "points" instead of 0.     |
//| Assumes HistorySelect() has already been called by the caller.    |
//+------------------------------------------------------------------+
double FindPositionOpenPrice(ulong positionId)
  {
   if(!HistorySelectByPosition(positionId)) return 0.0;
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong dt = HistoryDealGetTicket(i);
      if(dt == 0) continue;
      if((ENUM_DEAL_ENTRY)HistoryDealGetInteger(dt, DEAL_ENTRY) == DEAL_ENTRY_IN)
         return HistoryDealGetDouble(dt, DEAL_PRICE);
     }
   return 0.0;
  }

//+------------------------------------------------------------------+
//| AZZO STATE PERSISTENCE (GlobalVariable orqali)                    |
//| ------------------------------------------------------------------|
//| MUAMMO: timeframe o'zgartirilganda (masalan M5 -> M1) MetaTrader   |
//| avval OnDeinit(), keyin OnInit()ni QAYTA chaqiradi. OnInit() esa   |
//| barcha runtime (panel orqali tahrirlangan) qiymatlarni yana        |
//| Input* parametrlaridan o'qib, boshlang'ich holatga qaytarib        |
//| yuborar edi - shu sabab foydalanuvchi panelda 15 tani 19 taga      |
//| o'zgartirsa ham, timeframe almashtirganda yana 15 ga qaytardi.     |
//|                                                                     |
//| YECHIM: barcha panel orqali tahrirlanadigan qiymatlar (g_orderCount|
//| inputLots[], g_gridStep1/2, g_gridGap, g_rec*, g_nfpTime va h.k.)  |
//| har o'zgarganda MT5ning TERMINAL DARAJASIDAGI GlobalVariable       |
//| xotirasiga (GlobalVariableSet) yoziladi. GlobalVariable'lar        |
//| chartning timeframe'i, hatto EA qayta ishga tushishi bilan ham     |
//| O'CHMAYDI - faqat: (a) foydalanuvchi robotni CHARTDAN OLIB          |
//| TASHLAGANDA "HAMMASINI TOZALASH" tugmasi kabi maxsus tozalash       |
//| chaqirilsa, yoki (b) MT5 terminali o'zi juda uzoq vaqt              |
//| ishlatilmagan GlobalVariable'larni tozalasa, o'chadi.               |
//| OnInit() endi: 1) avval Input* dan DEFAULT qiymatlarni o'qiydi,     |
//| 2) keyin LoadAzzoState() orqali, agar oldin saqlangan holat         |
//| mavjud bo'lsa, ustidan yozadi. Shu bilan: robot BIRINCHI marta      |
//| chartga tashlanganda Input* qiymatlar ishlaydi, lekin undan keyin   |
//| (qayta compile qilinmaguncha, yoki robot to'liq olib tashlanib      |
//| qaytadan tashlanmaguncha) panelda qo'lda kiritilgan qiymatlar       |
//| DOIM saqlanib qoladi - timeframe necha marta almashtirilsa ham.     |
//+------------------------------------------------------------------+
string AzzoGVPrefix()
  {
   // --- Har bir SYMBOL uchun alohida holat saqlanadi (masalan GOLD va
   // --- EURUSD grafiklarida robot alohida-alohida ishlasa, bir-birining
   // --- sozlamalarini bosib ketmasin uchun) ---
   return "AZZO_NFP_PRO_" + _Symbol + "_";
  }

void SaveAzzoState()
  {
   string p = AzzoGVPrefix();

   GlobalVariableSet(p + "oc",     (double)g_orderCount);
   GlobalVariableSet(p + "gs1",    (double)g_gridStep1);
   GlobalVariableSet(p + "gs2",    (double)g_gridStep2);
   GlobalVariableSet(p + "gap",    (double)g_gridGap);
   GlobalVariableSet(p + "dist",   (double)g_orderDistance);
   GlobalVariableSet(p + "sl",     (double)g_orderSL);

   GlobalVariableSet(p + "rtrig",  (double)g_recTrigger);
   GlobalVariableSet(p + "roff",   (double)g_recOffset);
   GlobalVariableSet(p + "rstep",  (double)g_recStep);
   GlobalVariableSet(p + "rlvl",   (double)g_recLevels);
   GlobalVariableSet(p + "rlot",   g_recLot);
   GlobalVariableSet(p + "rblock", g_recBlockStopOrders ? 1.0 : 0.0);

   // --- g_nfpTime ("HH:MM:SS") GlobalVariable double formatiga
   // --- HHMMSS butun son sifatida kodlab yoziladi (masalan "13:30:05" -> 133005) ---
   int hh=0, mm=0, ss=0;
   string parts[];
   if(StringSplit(g_nfpTime, ':', parts) == 3)
     {
      hh = (int)StringToInteger(parts[0]);
      mm = (int)StringToInteger(parts[1]);
      ss = (int)StringToInteger(parts[2]);
     }
   GlobalVariableSet(p + "nfp", (double)(hh*10000 + mm*100 + ss));

   for(int i = 0; i < AZZO_MAX_GRID_ORDERS; i++)
      GlobalVariableSet(p + "lot" + (string)i, inputLots[i]);

   GlobalVariableSet(p + "saved", 1.0); // --- "holat mavjud" belgisi ---
  }

bool LoadAzzoState()
  {
   string p = AzzoGVPrefix();
   if(!GlobalVariableCheck(p + "saved")) return false; // --- hali hech qanday saqlangan holat yo'q (birinchi marta) ---

   g_orderCount = (int)GlobalVariableGet(p + "oc");
   if(g_orderCount < 1)  g_orderCount = 1;
   if(g_orderCount > AZZO_MAX_GRID_ORDERS) g_orderCount = AZZO_MAX_GRID_ORDERS;

   g_gridStep1 = (int)GlobalVariableGet(p + "gs1");
   g_gridStep2 = (int)GlobalVariableGet(p + "gs2");
   g_gridGap   = (int)GlobalVariableGet(p + "gap");
   if(GlobalVariableCheck(p + "dist")) g_orderDistance = (int)GlobalVariableGet(p + "dist");
   if(GlobalVariableCheck(p + "sl"))   g_orderSL       = (int)GlobalVariableGet(p + "sl");

   g_recTrigger = (int)GlobalVariableGet(p + "rtrig");
   g_recOffset  = (int)GlobalVariableGet(p + "roff");
   g_recStep    = (int)GlobalVariableGet(p + "rstep");
   g_recLevels  = (int)GlobalVariableGet(p + "rlvl");
   if(g_recLevels < 1)  g_recLevels = 1;
   if(g_recLevels > 50) g_recLevels = 50;
   g_recLot     = NormalizeLot(GlobalVariableGet(p + "rlot"));
   g_recBlockStopOrders = (GlobalVariableGet(p + "rblock") > 0.5);

   int nfpEncoded = (int)GlobalVariableGet(p + "nfp");
   int hh = nfpEncoded / 10000;
   int mm = (nfpEncoded / 100) % 100;
   int ss = nfpEncoded % 100;
   if(hh>=0 && hh<=23 && mm>=0 && mm<=59 && ss>=0 && ss<=59)
      g_nfpTime = StringFormat("%02d:%02d:%02d", hh, mm, ss);

   for(int i = 0; i < AZZO_MAX_GRID_ORDERS; i++)
     {
      string key = p + "lot" + (string)i;
      if(GlobalVariableCheck(key))
         inputLots[i] = NormalizeLot(GlobalVariableGet(key));
     }

   return true;
  }

//+------------------------------------------------------------------+
//| AZZO_ApplyPreset()                                                 |
//| MA'LUMOT (INFO) panelidagi "NFP", "CPI" va "OTHER NEWS" tugmalari  |
//| uchun YAGONA umumiy funksiya. Bosilgan tugmaga mos oldindan        |
//| belgilangan qiymatlarni BARCHA runtime (panel orqali tahrirlanadigan)|
//| o'zgaruvchilarga (g_orderCount, inputLots[], g_gridStep1/2, g_gridGap,|
//| g_orderDistance, g_orderSL) bir zumda yozadi, GlobalVariable orqali |
//| saqlaydi (timeframe almashtirilganda ham yo'qolmasin uchun) va      |
//| butun panelni (INFO, asosiy, RETRACEMENT, F.FACTORY - ochiq bo'lsa) |
//| DARHOL qayta chizadi - qayta compile qilish yoki EA'ni qayta ishga  |
//| tushirish SHART EMAS.                                               |
//|                                                                     |
//| lotFirst5  - 1,2,3,4,5-orderlarning HAR BIRIGA qo'llanadigan lot    |
//| lotRest    - 6-orderdan boshlab QOLGAN BARCHA darajalarga (6..50)   |
//|              qo'llanadigan lot (agar orderCount <= 5 bo'lsa, bu     |
//|              qiymat panelda hech qanday qatorga chiqmaydi, lekin    |
//|              xotirada tayyor turadi - zararsiz)                     |
//| step1      - 1-5 orderlar O'ZARO orasidagi masofa (PUNKT)           |
//| step2      - 6-50 orderlar O'ZARO orasidagi masofa (PUNKT)          |
//| gap        - 5-order bilan 6-order ORASIDAGI qo'shimcha masofa (PT) |
//| distance   - Order narxining Ask/Bid'dan uzoqligi (PUNKT)           |
//| sl         - Stop Loss masofasi (PUNKT)                             |
//+------------------------------------------------------------------+
void AZZO_ApplyPreset(int orderCount, double lotFirst5, double lotRest, int step1, int step2, int gap, int distance, int sl)
  {
   if(orderCount < 1) orderCount = 1;
   if(orderCount > AZZO_MAX_GRID_ORDERS) orderCount = AZZO_MAX_GRID_ORDERS;

   double safeLotFirst5 = NormalizeLot(lotFirst5);
   double safeLotRest   = NormalizeLot(lotRest);

   int individualCount = (orderCount < 5) ? orderCount : 5;
   for(int i = 0; i < individualCount; i++)
      inputLots[i] = safeLotFirst5;
   for(int i = individualCount; i < AZZO_MAX_GRID_ORDERS; i++)
      inputLots[i] = safeLotRest;

   g_orderCount    = orderCount;
   if(step1 < 0) step1 = 0;
   if(step2 < 0) step2 = 0;
   if(gap   < 0) gap   = 0;
   if(distance < 0) distance = 0;
   if(sl < 0) sl = 0;
   g_gridStep1     = step1;
   g_gridStep2     = step2;
   g_gridGap       = gap;
   g_orderDistance = distance;
   g_orderSL       = sl;

   SaveAzzoState(); // --- timeframe almashtirilsa ham preset o'chib ketmasin uchun saqlab qo'yamiz ---

   // --- Panel har doim INFO ochiq bo'lgan holatda ko'rinadi (chunki tugmalar faqat
   // --- INFO panel ichida) - shuning uchun CreateDashboard() + RelocatePanel() +
   // --- UpdateDashboard() orqali barcha maydonlar (Order soni, Lot qatorlari,
   // --- Grid Step, Masofa, SL) yangi qiymatlar bilan DARHOL qayta chiziladi.
   CreateDashboard();
   last_chart_w = -1;
   RelocatePanel();
   UpdateDashboard();
   if(newsPanelOpen) { UpdateNewsPanelTexts(); UpdateNewsCountdownLine(); }
   if(g_zonesVisible) DrawPendingZones(); // orderlar soni/masofa o'zgardi -> zonalar ham shu preset'ga moslashadi
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
//| "NFP" tugmasi preset qiymatlari                                    |
//| Orderlar soni: 15 | Order1-5: 0.02 Lot | 6-15: 0.01 Lot            |
//| 1-5 orasi: 70 PT | 6-50 orasi: 70 PT | 2 guruh orasi: 330 PT       |
//| Masofasi: 750 PT | SL: 750 PT                                      |
//+------------------------------------------------------------------+
void AZZO_ApplyPresetNFP()
  {
   AZZO_ApplyPreset(15, 0.02, 0.01, 70, 70, 330, 750, 750);
  }

//+------------------------------------------------------------------+
//| "CPI" tugmasi preset qiymatlari                                    |
//| Orderlar soni: 10 | Order1-5: 0.02 Lot | 6-10: 0.01 Lot            |
//| 1-5 orasi: 50 PT | 6-50 orasi: 50 PT | 2 guruh orasi: 220 PT       |
//| Masofasi: 500 PT | SL: 750 PT                                      |
//+------------------------------------------------------------------+
void AZZO_ApplyPresetCPI()
  {
   AZZO_ApplyPreset(10, 0.02, 0.01, 50, 50, 220, 500, 750);
  }

//+------------------------------------------------------------------+
//| "OTHER NEWS" tugmasi preset qiymatlari                             |
//| Orderlar soni: 5 | Order1-5: 0.02 Lot                              |
//| 1-5 orasi: 50 PT | 6-50 orasi: 50 PT | 2 guruh orasi: 220 PT       |
//| Masofasi: 500 PT | SL: 500 PT                                      |
//+------------------------------------------------------------------+
void AZZO_ApplyPresetOtherNews()
  {
   AZZO_ApplyPreset(5, 0.02, 0.01, 50, 50, 220, 500, 500);
  }

int OnInit()
  {
   trade.SetExpertMagicNumber(MAGIC_NUMBER);
   trade.SetDeviationInPoints(InpMaxSlippage); 
   trade.SetAsyncMode(true); // orderlar javobni kutmasdan yuboriladi -> panel/soat qotib qolmaydi
   ordersPlaced = false;
   globalTradeNFP = InpTradeNFP;
   currentCorner = (ENUM_PANEL_CORNER)InpPanelCorner; 

   // --- YANGI: "MT5 savdo tarixi" strelkalarini (qizil/ko'k, order ochilgan/yopilgan
   // --- joylarni belgilaydigan tarixiy belgilar) grafikda ko'rsatish/yashirish.
   // --- Bular panel/oynalarning OLDIGA chiqib ketib, ko'rinishni buzishi mumkin edi.
   // --- CHART_SHOW_TRADE_HISTORY - faqat vizual grafik sozlama, savdo logikasiga
   // --- (order ochish/yopish, straddle va h.k.) HECH QANDAY ta'sir qilmaydi.
   ChartSetInteger(0, CHART_SHOW_TRADE_HISTORY, !InpHideChartHistory);

   // --- Orderlar sonini (1-10) boshlang'ich holatda InpMaxOrders'dan olamiz,
   // --- keyinchalik esa faqat INFO panelidagi "Orderlar soni" maydoni orqali
   // --- (dastur qayta ishga tushirilmasdan) o'zgartirish mumkin bo'ladi.
   g_orderCount = (int)InpMaxOrders;
   if(g_orderCount < 1)  g_orderCount = 1;
   if(g_orderCount > AZZO_MAX_GRID_ORDERS) g_orderCount = AZZO_MAX_GRID_ORDERS;

   // --- RETRACEMENT (LIMIT ORDERS) paneli uchun runtime sozlamalarni Inp* qiymatlaridan
   // --- boshlang'ich holatga keltiramiz (keyin faqat panel orqali o'zgaradi) ---
   g_recTrigger = InpRecoveryTriggerPoints;
   g_recOffset  = InpRecoveryOffsetPoints;
   g_recStep    = InpRecoveryStepPoints;
   g_recLevels  = InpRecoveryLevels;
   if(g_recLevels < 1)  g_recLevels = 1;
   if(g_recLevels > 50) g_recLevels = 50;
   g_recLot     = NormalizeLot(InpRecoveryLot);

   // --- YANGI: "orderlar orasidagi masofa" (Grid Step) maydonlarini Inp* qiymatlaridan
   // --- boshlang'ich holatga keltiramiz (keyin faqat INFO paneli orqali o'zgaradi) ---
   g_gridStep1 = InpGridStepGroup1;
   g_gridStep2 = InpGridStepGroup2;
   g_gridGap   = InpGridGapBetweenGroups;

   // --- "MASOFASI" va "SL" INFO panel maydonlari uchun boshlang'ich holat ---
   g_orderDistance = InpDistance;
   g_orderSL       = InpStopLoss;

   // --- "HOLAT" tugmasi uchun boshlang'ich holat - InpRecoveryBlockStopOrders dan ---
   g_recBlockStopOrders = InpRecoveryBlockStopOrders;

   // --- NFP vaqtini input parametridan boshlang'ich holatga keltirish.
   // InpNewsTime ("HH:MM:SS") xato yoki diapazondan tashqari kiritilgan bo'lsa ham
   // EA ishga tushishda qulab tushmasligi uchun shu yerda ham tekshiramiz.
     {
      string normalized;
      int check = ParseNfpTimeInput(InpNewsTime, normalized);
      g_nfpTime = (check == 0) ? normalized : "15:29:55";
     }

   inputLots[0] = NormalizeLot(InpLotLevel1);
   inputLots[1] = NormalizeLot(InpLotLevel2);
   inputLots[2] = NormalizeLot(InpLotLevel3);
   inputLots[3] = NormalizeLot(InpLotLevel4);
   inputLots[4] = NormalizeLot(InpLotLevel5);
   inputLots[5] = NormalizeLot(InpLotLevel6);
   inputLots[6] = NormalizeLot(InpLotLevel7);
   inputLots[7] = NormalizeLot(InpLotLevel8);
   inputLots[8] = NormalizeLot(InpLotLevel9);
   inputLots[9] = NormalizeLot(InpLotLevel10);
   // --- 10 dan keyingi (11-50) darajalar uchun alohida input parametr yo'q -
   // --- boshlang'ich qiymat sifatida 0.01 qo'yiladi, keyin panel orqali
   // --- (guruhlangan Lot maydonlari yordamida) xohlagancha o'zgartirish mumkin.
   for(int lvlIdx = 10; lvlIdx < AZZO_MAX_GRID_ORDERS; lvlIdx++)
      inputLots[lvlIdx] = NormalizeLot(0.01);

   // --- AZZO STATE PERSISTENCE: agar bu robot shu SYMBOLda avval ishga
   // --- tushirilgan bo'lsa (masalan foydalanuvchi timeframe'ni
   // --- almashtirgani uchun OnInit qayta chaqirilgan bo'lsa), yuqoridagi
   // --- Input* qiymatlar o'rniga oxirgi PANEL orqali saqlangan qiymatlar
   // --- qo'llanadi. Robot birinchi marta chartga tashlanganda esa
   // --- LoadAzzoState() false qaytaradi va Input* qiymatlar shunday qoladi. ---
   LoadAzzoState();

   ApplyTheme(currentTheme);
   currentTargetTheme = currentTheme + 1;

   // --- AZZO ML: HMM bozor-rejim modelini tarixiy H1 barlaridan o'qitish.
   // --- Faqat shu yerda, bir marta chaqiriladi - robotning savdo logikasiga
   // --- (NFP straddle, grid, OCO) hech qanday ta'sir qilmaydi.
   AZZO_TrainRegimeModel();
   AZZO_UpdateRegimeIfDue();

   EventSetMillisecondTimer(50);

   // --- Panelni sichqoncha bilan sudrab olib borish (drag&drop) uchun
   // --- chartda "mouse move" hodisalarini yoqamiz. Bu bo'lmasa
   // --- CHARTEVENT_MOUSE_MOVE umuman kelmaydi.
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);

   CreateDashboard();
   last_chart_w = -1; 
   RelocatePanel();
   UpdateDashboard();

   // --- Pending Order Zones: agar InpZonesShowOnStart = true bo'lsa, EA ishga
   // --- tushishi bilanoq zonalar chizilgan holda ko'rinadi (aks holda "P"
   // --- yoki C tugmasi bosilgunicha yashirin turadi).
   g_zonesVisible = InpZonesShowOnStart;
   if(g_zonesVisible) DrawPendingZones();

   // --- Bid/Ask oralig'idagi KULRANG spread-zonasi "P"dan MUSTAQIL: robot chart'ga
   // --- ulangan zahoti (hozir, OnInit'da) darhol chiziladi va "P" bilan o'chirilmaydi.
   DrawSpreadZone();

   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(g_openBatchActive)  FlushOpenBatch();
   if(g_closeBatchActive) FlushCloseBatch();
   if(reason == REASON_REMOVE || reason == REASON_CHARTCLOSE)
     {
      ObjectsDeleteAll(0, prefix);
     }
   ChartRedraw(0);
  }

void OnTick()
  {
   CheckSmartOCO(); 
   ApplyTrailingStop();
   ManageRecoveryGrid();
   UpdateDashboard();

   // --- Bid/Ask o'zgarganda Pending Order Zones (agar ko'rinib turgan bo'lsa)
   // --- shu zahoti yangi narxga siljiydi. ObjectMove ishlatilgani uchun
   // --- (qayta yaratilmagani uchun) bu chaqiruv juda yengil, har tikda
   // --- xavfsiz ishlatilaveradi.
   if(g_zonesVisible) DrawPendingZones();

   // --- KULRANG spread-zonasi "P" holatidan MUSTAQIL - har doim, har tikda yangilanadi.
   DrawSpreadZone();
  }

void OnTimer()
  {
   // MUHIM: rang o'zgargandan so'ng DARHOL ChartRedraw() chaqiriladi. Aks holda
   // (avvalgi holat) rang o'zgarishlari darhol ekranga chiqmay, bir nechta
   // OnTimer tiklari to'planib, keyin birdaniga (sakrab) chiqardi - bu aynan
   // panelning "tebranib/silkinib" ko'rinishiga (shake effektiga) sabab bo'lgan.
   // Har bir tikda darhol redraw qilish orqali animatsiya silliq bo'ladi.
   if(rgbMode)
     {
      ApplySmoothRGB();
      ChartRedraw(0);
     }

   // --- Visual Button Hover FX: tugmalarning fon/border rangini har 50ms'da
   // --- maqsad holatga (hover/normal) bir qadam yaqinlashtiradi - shu orqali
   // --- sakramasdan, silliq rang o'tishi hosil bo'ladi.
   UpdateButtonHoverFX();

   // --- TESKARI SANOQ (Countdown) paneli - rang almashinuvi (sariq/orange/qizil)
   // va "BOOOM!" yozuvi aynan o'z vaqtida chiqishi uchun HAR OnTimer tikida (50ms)
   // yangilanadi - UpdateDashboard()/RelocatePanel() kabi 4 tikda bir marta emas.
   if(countdownPanelOpen)
     {
      UpdateCountdownPanel();
      ChartRedraw(0);
     }

   // --- Panel kichraytirilgan (minimized) bo'lsa - kichik satrdagi TESKARI
   // --- SANOQ ham countdownPanelOpen holatidan mustaqil, har 50ms'da yangilanadi.
   if(panelMinimized)
     {
      UpdateMinimizedCountdown();
      ChartRedraw(0);
     }

   // --- NFP OLDIN OGOHLANTIRISH: panel ochiq/yopiqligidan, minimallashtirilgan
   // --- bo'lishidan QAT'I NAZAR har doim ishlaydi - chunki bu MUHIM eslatma,
   // --- panel holatiga bog'liq bo'lmasligi kerak.
   CheckNfpPreAlert();

   // --- YANGILIKLAR KALENDARI (F) paneli - "keyingisigacha" sanoq (countdown)
   // HAR tikda (50ms) mahalliy hisoblanadi, tarmoqqa umuman bog'liq emas -
   // shuning uchun u DOIM soniyama-soniya, uzluksiz va HECH QACHON qotmasdan
   // yuradi. RefreshNewsCalendar() esa endi MT5'ning ICHKI Calendar API'sidan
   // (CalendarValueHistory) o'qiydi - bu 100% MAHALLIY (tarmoqsiz) chaqiruv,
   // hech qachon bloklanmaydi/qotmaydi, shuning uchun uni InpNewsRefreshSeconds
   // (standart 15 sek, xohlasa 1 sekundgacha tushirish mumkin) oralig'ida
   // xavfsiz ravishda tez-tez chaqirish mumkin.
   if(newsPanelOpen)
     {
      UpdateNewsCountdownLine();
      int newsRefreshMs = InpNewsRefreshSeconds * 1000;
      if(newsRefreshMs < 1000) newsRefreshMs = 1000; // minimal 1 sekund (mahalliy chaqiruv - xavfsiz)
      if(GetTickCount() - g_newsLastRefreshMs > (ulong)newsRefreshMs)
        {
         RefreshNewsCalendar();
         UpdateNewsPanelTexts();
         g_newsLastRefreshMs = GetTickCount();
        }
      ChartRedraw(0);
     }

   static int slowTicks = 0;
   slowTicks++;
   if(slowTicks >= 4) 
     {
      UpdateDashboard();
      RelocatePanel();
      slowTicks = 0;
     }

   // --- TELEGRAM BATCH: to'plamga so'nggi marta hodisa qo'shilganidan beri
   // --- BATCH_FLUSH_MS dan ko'proq vaqt o'tgan bo'lsa - endi yangi hodisa
   // --- kelmaydi deb hisoblab, to'plangan hammasini bitta xabar qilib yuboramiz.
   if(g_openBatchActive && (GetTickCount() - g_openBatchLastMs > BATCH_FLUSH_MS))
      FlushOpenBatch();
   if(g_closeBatchActive && (GetTickCount() - g_closeBatchLastMs > BATCH_FLUSH_MS))
      FlushCloseBatch();
   
   if(!globalTradeNFP || ordersPlaced) return;

   datetime brokerTime = TimeTradeServer();
   datetime curr = TimeCurrent();
   string date_str = TimeToString(curr, TIME_DATE);
   datetime targetTime = StringToTime(date_str + " " + g_nfpTime);

   if(brokerTime >= targetTime && brokerTime <= targetTime + InpTimeWindowSeconds)
     {
      PlaceStraddleOrders();
     }
  }

//+------------------------------------------------------------------+
//| OnTradeTransaction()                                               |
//| Yangi deal (order yopilishi/ochilishi) kelganda Telegram xabarini  |
//| tayyorlash uchun ishlatiladi.                                      |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                         const MqlTradeRequest      &request,
                         const MqlTradeResult       &result)
  {
   // --- TELEGRAM: yangi deal - faqat shu robotning o'z orderlari (MAGIC diapazoni) uchun ---
   if(InpTelegramEnabled && trans.type == TRADE_TRANSACTION_DEAL_ADD)
     {
      ulong dealTicket = trans.deal;
      if(HistoryDealSelect(dealTicket))
        {
         ulong dMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
         if(dMagic >= MAGIC_NUMBER && dMagic <= MAGIC_NUMBER + (ulong)g_orderCount)
           {
            ENUM_DEAL_TYPE dType = (ENUM_DEAL_TYPE)HistoryDealGetInteger(dealTicket, DEAL_TYPE);
            if(dType == DEAL_TYPE_BUY || dType == DEAL_TYPE_SELL)
              {
               ENUM_DEAL_ENTRY dEntry  = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
               double          dVolume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
               double          dPrice  = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
               string          typeStr = (dType == DEAL_TYPE_BUY) ? "BUY" : "SELL";

               if(dEntry == DEAL_ENTRY_IN && InpTelegramOnOpen)
                 {
                  // --- Ochilgan pozitsiyaning HOZIRGI (spread hisobidagi) foyda/zararini olamiz ---
                  ulong  posId       = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
                  double posProfit   = 0.0;
                  if(PositionSelectByTicket(posId))
                     posProfit = PositionGetDouble(POSITION_PROFIT);

                  int n = ArraySize(g_openBatchType);
                  ArrayResize(g_openBatchType,  n + 1);
                  ArrayResize(g_openBatchLot,   n + 1);
                  ArrayResize(g_openBatchPrice, n + 1);
                  ArrayResize(g_openBatchProfit,n + 1);
                  g_openBatchType[n]   = typeStr;
                  g_openBatchLot[n]    = dVolume;
                  g_openBatchPrice[n]  = dPrice;
                  g_openBatchProfit[n] = posProfit;

                  g_openBatchActive = true;
                  g_openBatchLastMs = GetTickCount();
                 }
               else if((dEntry == DEAL_ENTRY_OUT || dEntry == DEAL_ENTRY_OUT_BY) && InpTelegramOnClose)
                 {
                  double dProfit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT)
                                 + HistoryDealGetDouble(dealTicket, DEAL_SWAP)
                                 + HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);

                  ulong  posId     = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
                  double openPrice = FindPositionOpenPrice(posId);
                  double point     = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

                  // --- Position turi (BUY/SELL) closing deal turining teskarisi ---
                  string posTypeStr = (dType == DEAL_TYPE_SELL) ? "BUY" : "SELL";
                  int pointsResult = 0;
                  if(point > 0 && openPrice > 0)
                    {
                     if(posTypeStr == "BUY") pointsResult = (int)MathRound((dPrice - openPrice) / point);
                     else                    pointsResult = (int)MathRound((openPrice - dPrice) / point);
                    }

                  int n = ArraySize(g_closeBatchType);
                  ArrayResize(g_closeBatchType,      n + 1);
                  ArrayResize(g_closeBatchLot,       n + 1);
                  ArrayResize(g_closeBatchOpenPrice, n + 1);
                  ArrayResize(g_closeBatchClosePrice,n + 1);
                  ArrayResize(g_closeBatchProfit,    n + 1);
                  ArrayResize(g_closeBatchPoints,    n + 1);
                  g_closeBatchType[n]       = posTypeStr;
                  g_closeBatchLot[n]        = dVolume;
                  g_closeBatchOpenPrice[n]  = openPrice;
                  g_closeBatchClosePrice[n] = dPrice;
                  g_closeBatchProfit[n]     = dProfit;
                  g_closeBatchPoints[n]     = pointsResult;

                  g_closeBatchActive = true;
                  g_closeBatchLastMs = GetTickCount();
                 }
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| IsPointInsideObject()                                              |
//| Tester-compatibility helper: converts a chart object's corner-     |
//| relative XDISTANCE/YDISTANCE into absolute chart pixel coordinates |
//| and checks whether the given point (raw mouse click) falls inside. |
//+------------------------------------------------------------------+
bool IsPointInsideObject(string objName, int x, int y)
  {
   if(ObjectFind(0, objName) < 0) return false;

   long objX = ObjectGetInteger(0, objName, OBJPROP_XDISTANCE);
   long objY = ObjectGetInteger(0, objName, OBJPROP_YDISTANCE);
   long objW = ObjectGetInteger(0, objName, OBJPROP_XSIZE);
   long objH = ObjectGetInteger(0, objName, OBJPROP_YSIZE);
   ENUM_BASE_CORNER corner = (ENUM_BASE_CORNER)ObjectGetInteger(0, objName, OBJPROP_CORNER);

   int chartW = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int chartH = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);

   int absX = (int)objX;
   int absY = (int)objY;
   if(corner == CORNER_RIGHT_UPPER || corner == CORNER_RIGHT_LOWER) absX = chartW - (int)objX - (int)objW;
   if(corner == CORNER_LEFT_LOWER  || corner == CORNER_RIGHT_LOWER) absY = chartH - (int)objY - (int)objH;

   return (x >= absX && x <= absX + (int)objW && y >= absY && y <= absY + (int)objH);
  }

//+------------------------------------------------------------------+
//| InitHoverButtons()                                                |
//| "Visual Button Hover FX" ro'yxatini to'ldiradi: har bir haqiqiy   |
//| OBJ_BUTTON tugma uchun (1) hozirgi bazaviy fon/border rangi va    |
//| (2) sichqoncha ustiga kelganda erishiladigan neon "glow" rangi    |
//| belgilanadi. CreateDashboard() har safar chaqirilganda qayta      |
//| chaqiriladi (prefix o'zgarmasa ham, progresslarni 0'ga qaytarish  |
//| uchun) - shunda panel qayta chizilganda eski hover holati "yopishib|
//| qolmaydi".                                                        |
//+------------------------------------------------------------------+
void InitHoverButtons()
  {
   color cyanBase   = C'20,35,35';
   color cyanGlowBg = C'34,64,64';
   color redBase    = C'45,15,15';
   color redGlowBg  = C'80,24,24';

   int i = 0;
   g_hoverBtnName[i]=prefix+"btn_set";         g_hoverBaseBg[i]=cyanBase; g_hoverBaseBorder[i]=cyanBase; g_hoverGlowBg[i]=cyanGlowBg; g_hoverGlowBorder[i]=ColorMainNeon; i++;
   g_hoverBtnName[i]=prefix+"btn_cd_toggle";   g_hoverBaseBg[i]=cyanBase; g_hoverBaseBorder[i]=cyanBase; g_hoverGlowBg[i]=cyanGlowBg; g_hoverGlowBorder[i]=ColorMainNeon; i++;
   g_hoverBtnName[i]=prefix+"btn_retracement"; g_hoverBaseBg[i]=cyanBase; g_hoverBaseBorder[i]=cyanBase; g_hoverGlowBg[i]=cyanGlowBg; g_hoverGlowBorder[i]=ColorMainNeon; i++;
   g_hoverBtnName[i]=prefix+"btn_closeall";    g_hoverBaseBg[i]=redBase;  g_hoverBaseBorder[i]=redBase;  g_hoverGlowBg[i]=redGlowBg;  g_hoverGlowBorder[i]=clrRed; i++;
   g_hoverBtnName[i]=prefix+"btn_info";        g_hoverBaseBg[i]=redBase;  g_hoverBaseBorder[i]=redBase;  g_hoverGlowBg[i]=redGlowBg;  g_hoverGlowBorder[i]=clrRed; i++;
   g_hoverBtnName[i]=prefix+"btn_newstoggle";  g_hoverBaseBg[i]=redBase;  g_hoverBaseBorder[i]=redBase;  g_hoverGlowBg[i]=redGlowBg;  g_hoverGlowBorder[i]=clrRed; i++;
   // --- YANGI: MA'LUMOT (INFO) panelidagi "NFP" / "CPI" / "OTHER NEWS" preset tugmalari (cyan uslub) ---
   g_hoverBtnName[i]=prefix+"info_btn_nfp";        g_hoverBaseBg[i]=cyanBase; g_hoverBaseBorder[i]=cyanBase; g_hoverGlowBg[i]=cyanGlowBg; g_hoverGlowBorder[i]=ColorMainNeon; i++;
   g_hoverBtnName[i]=prefix+"info_btn_cpi";        g_hoverBaseBg[i]=cyanBase; g_hoverBaseBorder[i]=cyanBase; g_hoverGlowBg[i]=cyanGlowBg; g_hoverGlowBorder[i]=ColorMainNeon; i++;
   g_hoverBtnName[i]=prefix+"info_btn_othernews";  g_hoverBaseBg[i]=cyanBase; g_hoverBaseBorder[i]=cyanBase; g_hoverGlowBg[i]=cyanGlowBg; g_hoverGlowBorder[i]=ColorMainNeon; i++;

   for(int k = 0; k < HOVER_BTN_COUNT; k++)
     {
      g_hoverProgress[k] = 0.0;
      g_hoverTarget[k]   = false;
     }
   g_hoverBtnsRegistered = true;
  }

//+------------------------------------------------------------------+
//| UpdateButtonHoverHitTest()                                        |
//| CHARTEVENT_MOUSE_MOVE dan chaqiriladi (mouseX/mouseY - sichqoncha  |
//| piksel koordinatalari). Har bir ro'yxatdagi tugma uchun sichqoncha |
//| shu tugma ustida turgan-turmaganini aniqlab, "target" (maqsad)    |
//| holatini belgilaydi. Haqiqiy rang o'zgarishi (silliqlik) OnTimer'  |
//| dagi UpdateButtonHoverFX() ichida amalga oshadi.                  |
//+------------------------------------------------------------------+
void UpdateButtonHoverHitTest(int mouseX, int mouseY)
  {
   if(!g_hoverBtnsRegistered) return;
   for(int hb = 0; hb < HOVER_BTN_COUNT; hb++)
     {
      if(ObjectFind(0, g_hoverBtnName[hb]) < 0) continue;
      g_hoverTarget[hb] = IsPointInsideObject(g_hoverBtnName[hb], mouseX, mouseY);
     }
  }

//+------------------------------------------------------------------+
//| UpdateButtonHoverFX()                                             |
//| Har OnTimer tikida (50ms) chaqiriladi. Har bir tugma uchun progress|
//| qiymatini joriy holatdan ("target"ga qarab) bir qadam siljitadi va |
//| ColorBlend() yordamida bazaviy va "glow" ranglar orasida silliq    |
//| interpolyatsiya qilingan fon/border rangini qo'llaydi - natijada   |
//| tugma sichqoncha ustiga kelganda asta-sekin yorishadi (rang        |
//| o'zgarishi) va atrofida neon chiziq (border) paydo bo'ladi.        |
//+------------------------------------------------------------------+
void UpdateButtonHoverFX()
  {
   if(!g_hoverBtnsRegistered) return;

   double step = 0.22; // ~5 tik (taxminan 110ms) da to'liq o'tish - silliq, lekin sezilarli tez
   bool needRedraw = false;

   for(int hb = 0; hb < HOVER_BTN_COUNT; hb++)
     {
      if(ObjectFind(0, g_hoverBtnName[hb]) < 0) continue;

      double target = g_hoverTarget[hb] ? 1.0 : 0.0;
      if(MathAbs(g_hoverProgress[hb] - target) < 0.01)
        {
         if(g_hoverProgress[hb] != target)
           {
            g_hoverProgress[hb] = target;
           }
         else
           {
            continue; // allaqachon maqsadda - hech narsa qilinmaydi
           }
        }
      else if(g_hoverProgress[hb] < target)
        {
         g_hoverProgress[hb] += step;
         if(g_hoverProgress[hb] > target) g_hoverProgress[hb] = target;
        }
      else
        {
         g_hoverProgress[hb] -= step;
         if(g_hoverProgress[hb] < target) g_hoverProgress[hb] = target;
        }

      double pct = g_hoverProgress[hb] * 100.0; // ColorBlend foiz (0..100) kutadi
      color bg     = ColorBlend(g_hoverGlowBg[hb],     g_hoverBaseBg[hb],     pct);
      color border = ColorBlend(g_hoverGlowBorder[hb], g_hoverBaseBorder[hb], pct);

      ObjectSetInteger(0, g_hoverBtnName[hb], OBJPROP_BGCOLOR, bg);
      ObjectSetInteger(0, g_hoverBtnName[hb], OBJPROP_BORDER_COLOR, border);
      needRedraw = true;
     }

   if(needRedraw) ChartRedraw(0);
  }

//+------------------------------------------------------------------+
//| HandlePanelButtonClick()                                          |
//| Single source of truth for "a panel button named <btnName> was    |
//| clicked". Called both from CHARTEVENT_OBJECT_CLICK (normal live   |
//| chart behavior, unchanged) AND from the CHARTEVENT_CLICK hit-test  |
//| fallback below (needed because the Strategy Tester's visual-mode   |
//| chart does not always deliver CHARTEVENT_OBJECT_CLICK for          |
//| OBJ_BUTTON objects). btnName already includes the "prefix".        |
//+------------------------------------------------------------------+
void HandlePanelButtonClick(string btnName)
  {
   if(btnName == prefix+"btn_minimize" || btnName == prefix+"btn_min_txt")
     {
      panelMinimized = !panelMinimized;
      if(panelMinimized) { infoPanelOpen = false; countdownPanelOpen = false; newsPanelOpen = false; }
      CreateDashboard(); 
      last_chart_w = -1; 
      RelocatePanel();
      ChartRedraw(0);
     }
   if(btnName == prefix+"btn_info")
     {
      infoPanelOpen = !infoPanelOpen; // Boshqa panellarni majburiy yopmaydi
      CreateDashboard();
      last_chart_w = -1;
      RelocatePanel();
      UpdateDashboard();
      // --- MUHIM: CreateDashboard() F.FACTORY paneli ochiq bo'lsa ham uning
      // --- barcha matn obyektlarini "Yuklanmoqda..." holatida QAYTADAN yaratadi.
      // --- Shu sabab, agar F.FACTORY ochiq bo'lsa, allaqachon xotirada turgan
      // --- g_news* ma'lumotlarini DARHOL qayta chizamiz - shunda u "yana
      // --- yuklanayotgandek" ko'rinmaydi va hech qanday tarmoq so'rovi
      // --- YUBORILMAYDI (faqat mavjud ma'lumot qayta chiziladi).
      if(newsPanelOpen)
        {
         UpdateNewsPanelTexts();
         UpdateNewsCountdownLine();
        }
      ChartRedraw(0);
     }
   if(btnName == prefix+"btn_newstoggle")
     {
      newsPanelOpen = !newsPanelOpen; // Asosiy paneldagi "F.FACTORY [ F ]" tugmasi - "F" tez tugmasi bilan bir xil ishlaydi
      CreateDashboard();
      last_chart_w = -1;
      RelocatePanel();
      UpdateDashboard(); // --- YANGI: asosiy panel bilan TO'LIQ sinxron (avval yetishmayotgan edi) ---
      if(newsPanelOpen)
        {
         // --- YANGI: panel DARHOL ("Yuklanmoqda...") ko'rinsin, shundan keyin
         // --- og'ir CalendarValueHistory() hisob-kitobi ishga tushsin - shunda
         // --- foydalanuvchi hech narsa ko'rinmay turib kutmaydi, panel TEZ chiqadi.
         ChartRedraw(0);
         RefreshNewsCalendar();
         g_newsLastRefreshMs = GetTickCount();
        }
      UpdateNewsPanelTexts();
      UpdateNewsCountdownLine();
      ChartRedraw(0);
     }
   if(btnName == prefix+"btn_cd_toggle")
     {
      countdownPanelOpen = !countdownPanelOpen;
      CreateDashboard();
      last_chart_w = -1;
      RelocatePanel();
      UpdateDashboard();
      UpdateCountdownPanel();
      // --- Xuddi INFO tugmasidagi kabi: F.FACTORY ochiq bo'lsa, darhol qayta chizamiz ---
      if(newsPanelOpen)
        {
         UpdateNewsPanelTexts();
         UpdateNewsCountdownLine();
        }
      ChartRedraw(0);
     }
   if(btnName == prefix+"btn_retracement")
     {
      retracementPanelOpen = !retracementPanelOpen; // "RETRACEMENT [ R ]" tugmasi - "R" tez tugmasi bilan bir xil ishlaydi
      CreateDashboard();
      last_chart_w = -1;
      RelocatePanel();
      UpdateDashboard();
      // --- Boshqa yon panellar (INFO/F.FACTORY) ochiq bo'lsa, darhol qayta chizamiz ---
      if(newsPanelOpen)
        {
         UpdateNewsPanelTexts();
         UpdateNewsCountdownLine();
        }
      ChartRedraw(0);
     }
   // --- YANGI: "HOLAT" (TRUE/FALSE) kichik tugmasi - LIMIT ORDERS panelining
   // --- eng tepasida. Har bosilganda g_recBlockStopOrders TRUE <-> FALSE
   // --- almashadi va tugma matni/rangi darhol yangilanadi - qayta compile
   // --- yoki butun panelni qayta chizish (CreateDashboard) shart emas.
   if(btnName == prefix+"retr_val_status")
     {
      g_recBlockStopOrders = !g_recBlockStopOrders;
      SaveAzzoState();
      string newText = g_recBlockStopOrders ? "TRUE" : "FALSE";
      color  newCol  = g_recBlockStopOrders ? C'0,255,102' : clrRed;
      ObjectSetString(0, prefix+"retr_val_status", OBJPROP_TEXT, newText);
      ObjectSetInteger(0, prefix+"retr_val_status", OBJPROP_COLOR, newCol);
      ObjectSetInteger(0, prefix+"retr_val_status", OBJPROP_BORDER_COLOR, newCol);
      ObjectSetInteger(0, prefix+"retr_val_status", OBJPROP_STATE, false);
      ChartRedraw(0);
     }
   if(!panelMinimized)
     {
      if(btnName == prefix+"btn_set")
        {
         PlaceManualOrders();
         ObjectSetInteger(0, prefix+"btn_set", OBJPROP_STATE, false);
         ChartRedraw(0);
        }
      if(btnName == prefix+"btn_closeall")
        {
         CloseAllPositionsAndOrders();
         ObjectSetInteger(0, prefix+"btn_closeall", OBJPROP_STATE, false);
         ChartRedraw(0);
        }

      // --- YANGI: MA'LUMOT (INFO) panelidagi "NFP" / "CPI" / "OTHER NEWS" preset
      // --- tugmalari. Bosilgan zahoti tegishli AZZO_ApplyPreset*() funksiyasi
      // --- chaqiriladi - u o'zi CreateDashboard()/RelocatePanel()/UpdateDashboard()
      // --- orqali panelni darhol yangi qiymatlar bilan qayta chizadi, shuning uchun
      // --- bu yerda qo'shimcha ChartRedraw() shart emas (funksiya ichida bor).
      if(btnName == prefix+"info_btn_nfp")
        {
         AZZO_ApplyPresetNFP();
         if(ObjectFind(0, prefix+"info_btn_nfp") >= 0) ObjectSetInteger(0, prefix+"info_btn_nfp", OBJPROP_STATE, false);
        }
      if(btnName == prefix+"info_btn_cpi")
        {
         AZZO_ApplyPresetCPI();
         if(ObjectFind(0, prefix+"info_btn_cpi") >= 0) ObjectSetInteger(0, prefix+"info_btn_cpi", OBJPROP_STATE, false);
        }
      if(btnName == prefix+"info_btn_othernews")
        {
         AZZO_ApplyPresetOtherNews();
         if(ObjectFind(0, prefix+"info_btn_othernews") >= 0) ObjectSetInteger(0, prefix+"info_btn_othernews", OBJPROP_STATE, false);
        }
     }
  }

void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   // --- PANELNI SICHQONCHA BILAN SUDRASH (DRAG & DROP) -------------------------
   // Header (sarlavha) qismidan (yoki minimallashtirilgan holatda min_bg dan)
   // chap tugma bosilib ushlab turilsa va sichqoncha siljitilsa, panel
   // xohlagan joyga erkin ko'chib boradi. Tugma qo'yib yuborilganda joy
   // saqlanib qoladi (g_manualPos = true), shundan keyin panel avtomatik
   // burchak (corner) hisobiga emas, balki shu saqlangan joyga chiqadi.
   if(id == CHARTEVENT_MOUSE_MOVE)
     {
      int mouseX = (int)lparam;
      int mouseY = (int)dparam;
      int btnState = (int)StringToInteger(sparam); // bit 1 = chap tugma bosilgan
      bool leftDown = ((btnState & 1) == 1);

      string dragHandle = panelMinimized ? (prefix+"min_bg") : (prefix+"header");

      // --- Visual Button Hover FX: har mouse-move hodisasida tugmalar ustida
      // --- sichqoncha bor-yo'qligini tekshiramiz; silliq rang o'tishi
      // --- OnTimer()dagi UpdateButtonHoverFX() orqali amalga oshiriladi.
      // --- MUHIM (performance fix): sudrash faol bo'lganda bu tekshiruvni
      // --- o'tkazib yuboramiz - drag paytida tugma hover'i baribir
      // --- ko'rinmaydi/kerak emas, lekin har hodisada 6 ta ObjectFind +
      // --- 24 ta ObjectGetInteger chaqirig'i qo'shimcha yuk beradi va
      // --- panelning "orqada qolib sudralishi"ga hissa qo'shadi.
      if(!g_dragActive)
         UpdateButtonHoverHitTest(mouseX, mouseY);

      if(leftDown)
        {
         if(!g_dragActive)
           {
            // Faqat sarlavha (header) ustida chap tugma bosilsa sudrash boshlanadi -
            // shunda tugmalar, edit maydonlari va boshqa elementlar bosilishiga xalaqit bermaydi.
            if(IsPointInsideObject(dragHandle, mouseX, mouseY))
              {
               g_dragActive      = true;
               g_dragStartMouseX = mouseX;
               g_dragStartMouseY = mouseY;
               g_dragStartPanelX = (int)ObjectGetInteger(0, dragHandle, OBJPROP_XDISTANCE);
               g_dragStartPanelY = (int)ObjectGetInteger(0, dragHandle, OBJPROP_YDISTANCE);
               ChartSetInteger(0, CHART_MOUSE_SCROLL, false); // sudrash paytida chart siljib ketmasin
               g_lastDragRedrawMs = 0; // birinchi harakatda darhol chizilishi uchun
              }
           }
         else
           {
            int newX = g_dragStartPanelX + (mouseX - g_dragStartMouseX);
            int newY = g_dragStartPanelY + (mouseY - g_dragStartMouseY);

            g_panelX    = newX;
            g_panelY    = newY;
            g_manualPos = true;

            // --- MUHIM (performance fix): RelocatePanel()+ChartRedraw() har bir
            // --- MOUSE_MOVE hodisasida emas, balki max ~60 fps (16ms) da bir marta
            // --- chaqiriladi. Aks holda terminal juda tez-tez yuborayotgan
            // --- sichqoncha hodisalarini navbatga to'plab qo'yadi (chunki har
            // --- safar 30-90 ta obyekt qayta joylashtirilib, qayta chizilyapti),
            // --- va panel sichqonchadan orqada qolib, keyin "sakrab" quvib yetadi -
            // --- aynan sizga yoqmagan kechikish/qotish shu sababdan edi.
            uint nowMs = GetTickCount();
            if(nowMs - g_lastDragRedrawMs >= 16)
              {
               g_lastDragRedrawMs = nowMs;
               RelocatePanel(true); // force = true -> darhol yangi joyga ko'chiradi
               ChartRedraw(0);
              }
           }
        }
      else
        {
         if(g_dragActive)
           {
            g_dragActive = false;
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true); // sudrash tugadi - chartni qayta yoqamiz
            // --- Sichqoncha qo'yib yuborilgandan keyin, oxirgi throttle oralig'ida
            // --- "yutib qolingan" harakat bo'lsa ham, panel ANIQ oxirgi joyga
            // --- (g_panelX/g_panelY) to'liq sinxronlanishi uchun yakuniy relocate.
            RelocatePanel(true);
            ChartRedraw(0);
           }
        }
      return;
     }

   if(id == CHARTEVENT_CHART_CHANGE)
     {
      if(ObjectFind(0, prefix+"bg") < 0 && ObjectFind(0, prefix+"min_bg") < 0) CreateDashboard();
      last_chart_w = -1; 
      RelocatePanel();
      UpdateDashboard();
      if(newsPanelOpen) { UpdateNewsPanelTexts(); UpdateNewsCountdownLine(); }
      ChartRedraw(0);
     }
     
   if(id == CHARTEVENT_OBJECT_CLICK)
     {
      HandlePanelButtonClick(sparam);
     }

   // --- MA'LUMOT panelidagi tahrirlanadigan Lot maydonlari (info_val_0 ... info_val_9) ---
   // Foydalanuvchi Edit box ichiga yangi lot kiritib, undan tashqariga bosgan yoki Enter
   // bosgan zahoti CHARTEVENT_OBJECT_ENDEDIT hodisasi keladi. Bu yerda kiritilgan qiymat
   // brokerning min/max/lot-step chegaralariga moslab (NormalizeLot orqali) tekshiriladi
   // va tasdiqlangan qiymat qaytadan Edit box ichiga yoziladi.
   if(id == CHARTEVENT_OBJECT_ENDEDIT)
     {
      // --- INFO paneldagi Lot qatorlari endi GURUHLANGAN bo'lishi mumkin (masalan
      // --- "6-10 :" bitta qator bo'lib, unga kiritilgan lot shu diapazondagi
      // --- BARCHA orderlarga baravariga qo'llanadi). Qaysi qator qaysi order(lar)ga
      // --- mos kelishini BuildInfoRows() joriy g_orderCount asosida hisoblab beradi.
        {
         int rowStart[]; int rowEnd[];
         int rowCount = BuildInfoRows(g_orderCount, rowStart, rowEnd);

         for(int i = 0; i < rowCount; i++)
           {
            string editName = prefix + "info_val_" + (string)i;
            if(sparam == editName)
              {
               string rawText  = ObjectGetString(0, editName, OBJPROP_TEXT);
               double rawLot   = StringToDouble(rawText);
               double safeLot  = NormalizeLot(rawLot);

               for(int k = rowStart[i]; k <= rowEnd[i]; k++)
                  inputLots[k] = safeLot;

               ObjectSetString(0, editName, OBJPROP_TEXT, DoubleToString(safeLot, 2));
               SaveAzzoState(); // --- panel orqali o'zgartirilgan lot(lar)ni saqlab qo'yamiz ---
               ChartRedraw(0);
               break;
              }
           }
        }

      // --- "Orderlar soni" (info_ordercount) maydoni: 1 dan AZZO_MAX_GRID_ORDERS
      // gacha bo'lgan butun son. Kiritilgan qiymat shu diapazondan tashqariga
      // chiqsa, avtomatik ravishda eng yaqin chegaraga qisqartiriladi. Qiymat
      // o'zgarganda panel qaytadan chiziladi - chunki INFO paneldagi lot qatorlari
      // soni ham shunga mos ravishda ko'payadi/kamayadi (1-5 alohida, qolgani
      // 5 tadan guruhlangan), va keyingi PlaceStraddleOrders() chaqiruvi ham aynan
      // shu sonda BUY STOP + SELL STOP qo'yadi.
      if(sparam == prefix + "info_ordercount")
        {
         string rawText = ObjectGetString(0, prefix + "info_ordercount", OBJPROP_TEXT);
         int newCount = (int)StringToInteger(rawText);
         if(newCount < 1)  newCount = 1;
         if(newCount > AZZO_MAX_GRID_ORDERS) newCount = AZZO_MAX_GRID_ORDERS;
         g_orderCount = newCount;
         SaveAzzoState(); // --- yangi "Orderlar soni"ni saqlab qo'yamiz (timeframe almashsa ham yo'qolmasin) ---

         CreateDashboard();
         last_chart_w = -1;
         RelocatePanel();
         UpdateDashboard();
         if(newsPanelOpen) { UpdateNewsPanelTexts(); UpdateNewsCountdownLine(); }
         if(g_zonesVisible) DrawPendingZones(); // orderlar soni o'zgardi -> zonalar ham shu songa moslashadi
         ChartRedraw(0);
        }

      // --- YANGI: INFO paneldagi "ORDERLAR ORASI" (Grid Step) uchta maydoni -----
      // (info_grid_val_1 = 1-5 orasi, info_grid_val_2 = 6-50 orasi,
      //  info_grid_val_3 = 2 guruh orasi). Har biri PUNKTDA butun son, 0 dan kichik
      // bo'lishi mumkin emas. Qiymat darhol g_gridStep1/g_gridStep2/g_gridGap ga
      // yoziladi va keyingi GetGridOffsetPoints() chaqiruvida (qayta compile
      // qilmasdan) ishlatiladi - xuddi RETRACEMENT paneli maydonlari kabi.
      if(sparam == prefix + "info_grid_val_1")
        {
         int v = (int)StringToInteger(ObjectGetString(0, prefix+"info_grid_val_1", OBJPROP_TEXT));
         if(v < 0) v = 0;
         g_gridStep1 = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"info_grid_val_1", OBJPROP_TEXT, (string)v);
         ChartRedraw(0);
        }
      if(sparam == prefix + "info_grid_val_2")
        {
         int v = (int)StringToInteger(ObjectGetString(0, prefix+"info_grid_val_2", OBJPROP_TEXT));
         if(v < 0) v = 0;
         g_gridStep2 = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"info_grid_val_2", OBJPROP_TEXT, (string)v);
         ChartRedraw(0);
        }
      if(sparam == prefix + "info_grid_val_3")
        {
         int v = (int)StringToInteger(ObjectGetString(0, prefix+"info_grid_val_3", OBJPROP_TEXT));
         if(v < 0) v = 0;
         g_gridGap = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"info_grid_val_3", OBJPROP_TEXT, (string)v);
         ChartRedraw(0);
        }

      // --- YANGI: "MASOFASI" (info_grid_val_4) va "SL" (info_grid_val_5) maydonlari -
      // --- Order narxining Ask/Bid'dan uzoqligi va Stop Loss masofasi (PUNKTDA).
      // --- Qiymat darhol g_orderDistance/g_orderSL ga yoziladi va keyingi
      // --- PlaceStraddleOrders() / DrawPendingZones() chaqiruvida (qayta compile
      // --- qilmasdan) ishlatiladi. 0 dan kichik bo'lishi mumkin emas.
      if(sparam == prefix + "info_grid_val_4")
        {
         int v = (int)StringToInteger(ObjectGetString(0, prefix+"info_grid_val_4", OBJPROP_TEXT));
         if(v < 0) v = 0;
         g_orderDistance = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"info_grid_val_4", OBJPROP_TEXT, (string)v);
         ChartRedraw(0);
        }
      if(sparam == prefix + "info_grid_val_5")
        {
         int v = (int)StringToInteger(ObjectGetString(0, prefix+"info_grid_val_5", OBJPROP_TEXT));
         if(v < 0) v = 0;
         g_orderSL = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"info_grid_val_5", OBJPROP_TEXT, (string)v);
         ChartRedraw(0);
        }

      // --- RETRACEMENT (LIMIT ORDERS) panelidagi 5 ta sozlama maydoni (retr_val_0..4).
      // Har biri o'zining chegarasiga moslab tekshiriladi va tegishli g_rec*
      // o'zgaruvchisiga yoziladi - keyingi PlaceRecoveryGrid() chaqiruvi darhol
      // (qayta compile qilmasdan) shu yangi qiymatlar bilan ishlaydi.
      if(sparam == prefix + "retr_val_0") // TRIGGER (PUNKT)
        {
         int v = (int)StringToInteger(ObjectGetString(0, prefix+"retr_val_0", OBJPROP_TEXT));
         if(v < 0) v = 0;
         g_recTrigger = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"retr_val_0", OBJPROP_TEXT, (string)v);
         ChartRedraw(0);
        }
      if(sparam == prefix + "retr_val_1") // 1-DARAJA MASOFASI (PUNKT)
        {
         int v = (int)StringToInteger(ObjectGetString(0, prefix+"retr_val_1", OBJPROP_TEXT));
         if(v < 0) v = 0;
         g_recOffset = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"retr_val_1", OBJPROP_TEXT, (string)v);
         ChartRedraw(0);
        }
      if(sparam == prefix + "retr_val_2") // QADAM (PUNKT)
        {
         int v = (int)StringToInteger(ObjectGetString(0, prefix+"retr_val_2", OBJPROP_TEXT));
         if(v < 1) v = 1;
         g_recStep = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"retr_val_2", OBJPROP_TEXT, (string)v);
         ChartRedraw(0);
        }
      if(sparam == prefix + "retr_val_3") // ORDERLAR SONI (har tomonda) - 1 dan 50 gacha
        {
         int v = (int)StringToInteger(ObjectGetString(0, prefix+"retr_val_3", OBJPROP_TEXT));
         if(v < 1)  v = 1;
         if(v > 50) v = 50;
         g_recLevels = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"retr_val_3", OBJPROP_TEXT, (string)v);
         ChartRedraw(0);
        }
      if(sparam == prefix + "retr_val_4") // LOT
        {
         double v = NormalizeLot(StringToDouble(ObjectGetString(0, prefix+"retr_val_4", OBJPROP_TEXT)));
         g_recLot = v;
         SaveAzzoState();
         ObjectSetString(0, prefix+"retr_val_4", OBJPROP_TEXT, DoubleToString(v, 2));
         ChartRedraw(0);
        }

      // --- YANGI: "NFP VAQTI" (val_trig) maydoni -------------------------------
      // Foydalanuvchi panel ustida to'g'ridan-to'g'ri "HH:MM:SS" formatida yangi
      // NFP vaqtini kiritadi (masalan "13:30:00"). Qayta compile shart emas.
      //
      //  - Format noto'g'ri bo'lsa (raqamdan boshqa belgi, ":" tuzilishi buzilgan,
      //    masalan "111:11:!11") -> ESKI (joriy) qiymat saqlanib qoladi, maydon
      //    o'z holiga qaytadi.
      //  - Raqamlar to'g'ri, lekin 00:00:00-23:59:59 diapazonidan tashqari bo'lsa
      //    (masalan "24:00:00") -> standart "15:29:55" ga qaytariladi.
      //  - Hammasi to'g'ri bo'lsa -> yangi vaqt darhol qabul qilinadi va
      //    keyingi OnTimer() tekshiruvida ishlatiladi.
      if(sparam == prefix + "val_trig")
        {
         string rawTimeText = ObjectGetString(0, prefix + "val_trig", OBJPROP_TEXT);
         string normalizedTime;
         int    parseResult = ParseNfpTimeInput(rawTimeText, normalizedTime);

         if(parseResult == 0)
           {
            if(normalizedTime != g_nfpTime) ordersPlaced = false; // yangi vaqt -> qayta tayyor
            g_nfpTime = normalizedTime;              // to'g'ri kiritildi
           }
         else if(parseResult == 2)
           {
            if(g_nfpTime != "15:29:55") ordersPlaced = false;
            g_nfpTime = "15:29:55";                  // diapazondan tashqari -> standart qiymat
           }

         // parseResult == 1 (format xatosi) bo'lsa g_nfpTime o'zgarishsiz qoladi.
         if(parseResult == 0 || parseResult == 2) SaveAzzoState();
         ObjectSetString(0, prefix + "val_trig", OBJPROP_TEXT, g_nfpTime);
         ChartRedraw(0);
        }
     }

   // --- Strategy Tester visual-mode fallback ------------------------------------
   // Raw mouse clicks (CHARTEVENT_CLICK) are delivered reliably in the tester even
   // when CHARTEVENT_OBJECT_CLICK is not. We hit-test the click's pixel coordinates
   // against every clickable panel object; on a match we run the EXACT same handler
   // used above, so behavior is identical on a live chart and inside the tester.
   //
   // MUHIM TUZATISH: bu blok FAQAT Strategy Tester ichida ishlashi kerak edi.
   // MQL_TESTER tekshiruvi yo'q bo'lgani sabab, jonli/demo/real chartda HAR bir
   // tugma bosilganda CHARTEVENT_OBJECT_CLICK (yuqorida) VA shu CHARTEVENT_CLICK
   // ikkalasi ham ishga tushib, HandlePanelButtonClick() ikki marta chaqirilar
   // edi - shuning uchun panel bir bosishda "ochilib, darhol yopilib" qolardi.
   // Endi bu blok faqat testerda (MQL_TESTER=true) ishlaydi, jonli chartda esa
   // faqat CHARTEVENT_OBJECT_CLICK (bitta marta) ishlaydi.
   if(id == CHARTEVENT_CLICK && MQLInfoInteger(MQL_TESTER))
     {
      int clickX = (int)lparam;
      int clickY = (int)dparam;

      string clickable[] =
        {
         "btn_minimize", "btn_min_txt", "btn_info", "btn_newstoggle", "btn_cd_toggle", "btn_retracement",
         "btn_set", "btn_closeall", "info_btn_nfp", "info_btn_cpi", "info_btn_othernews"
        };

      for(int i = 0; i < ArraySize(clickable); i++)
        {
         string fullName = prefix + clickable[i];
         if(IsPointInsideObject(fullName, clickX, clickY))
           {
            HandlePanelButtonClick(fullName);
            break;
           }
        }
     }
     
   if(id == CHARTEVENT_KEYDOWN)
     {
      if(lparam == 73) // I
        {
         infoPanelOpen = !infoPanelOpen;
         CreateDashboard();
         last_chart_w = -1;
         RelocatePanel();
         UpdateDashboard();
         // --- F.FACTORY ochiq bo'lsa, darhol qayta chizamiz (tarmoq so'rovisiz) ---
         if(newsPanelOpen)
           {
            UpdateNewsPanelTexts();
            UpdateNewsCountdownLine();
           }
         ChartRedraw(0);
         return;
        }
        
      if(lparam == 84) // T -> TESKARI SANOQ (Countdown) panelini ochish/yopish
        {
         countdownPanelOpen = !countdownPanelOpen;
         CreateDashboard();
         last_chart_w = -1;
         RelocatePanel();
         UpdateDashboard();
         UpdateCountdownPanel();
         // --- F.FACTORY ochiq bo'lsa, darhol qayta chizamiz (tarmoq so'rovisiz) ---
         if(newsPanelOpen)
           {
            UpdateNewsPanelTexts();
            UpdateNewsCountdownLine();
           }
         ChartRedraw(0);
         return;
        }

      if(lparam == 70) // F -> YANGILIKLAR KALENDARI panelini ochish/yopish
        {
         newsPanelOpen = !newsPanelOpen;
         CreateDashboard();
         last_chart_w = -1;
         RelocatePanel();
         UpdateDashboard(); // --- YANGI: asosiy panel bilan TO'LIQ sinxron (avval yetishmayotgan edi) ---
         if(newsPanelOpen)
           {
            // --- YANGI: panel DARHOL ("Yuklanmoqda...") ko'rinsin, shundan keyin
            // --- og'ir CalendarValueHistory() hisob-kitobi ishga tushsin - shunda
            // --- panel TEZROQ chiqqandek his qilinadi.
            ChartRedraw(0);
            RefreshNewsCalendar();
            g_newsLastRefreshMs = GetTickCount();
           }
         UpdateNewsPanelTexts();
         UpdateNewsCountdownLine();
         ChartRedraw(0);
         return;
        }

      if(lparam == 82) // R -> RETRACEMENT (LIMIT ORDERS) panelini ochish/yopish
        {
         retracementPanelOpen = !retracementPanelOpen;
         CreateDashboard();
         last_chart_w = -1;
         RelocatePanel();
         UpdateDashboard();
         // --- F.FACTORY ochiq bo'lsa, darhol qayta chizamiz (tarmoq so'rovisiz) ---
         if(newsPanelOpen)
           {
            UpdateNewsPanelTexts();
            UpdateNewsCountdownLine();
           }
         ChartRedraw(0);
         return;
        }


      if(lparam == 83) // S -> Panelni Kichraytirish/Kattalashtirish (xuddi "-"/"+" tugmasi bosilgandek)
        {
         HandlePanelButtonClick(prefix+"btn_minimize");
         return;
        }

      // --- P -> Pending Order Zones (Buy/Sell Stop, SL, TP chiziqlari) ON/OFF.
      // Bir marta bossa - chiziladi (yonadi), yana bir marta bossa - g'oyib bo'ladi.
      // Hotkeys o'chirilgan (InpUseHotkeys=false) bo'lsa ham ishlaydi - I/T/F/R/S kabi.
      if(lparam == 80) // P
        {
         g_zonesVisible = !g_zonesVisible;
         if(g_zonesVisible) DrawPendingZones();
         else               DeleteAllZoneLines();
         ChartRedraw(0);
         return;
        }

      if(!InpUseHotkeys) return;

      bool shiftDown = (TerminalInfoInteger(TERMINAL_KEYSTATE_SHIFT) < 0);
      bool ctrlDown  = (TerminalInfoInteger(TERMINAL_KEYSTATE_CONTROL) < 0);

      // --- Ctrl + Arrow keys: move the panel to a different chart corner --------
      if(ctrlDown)
        {
         bool cornerChanged = true;
         bool isRight = (currentCorner == Corner_Right_Upper || currentCorner == Corner_Right_Lower);
         bool isLower = (currentCorner == Corner_Left_Lower  || currentCorner == Corner_Right_Lower);

         if(lparam == 38) // Ctrl + UP
           {
            currentCorner = isRight ? Corner_Right_Upper : Corner_Left_Upper;
           }
         else if(lparam == 40) // Ctrl + DOWN
           {
            currentCorner = isRight ? Corner_Right_Lower : Corner_Left_Lower;
           }
         else if(lparam == 37) // Ctrl + LEFT
           {
            currentCorner = isLower ? Corner_Left_Lower : Corner_Left_Upper;
           }
         else if(lparam == 39) // Ctrl + RIGHT
           {
            currentCorner = isLower ? Corner_Right_Lower : Corner_Right_Upper;
           }
         else
           {
            cornerChanged = false;
           }

         if(cornerChanged)
           {
            g_manualPos = false; // qo'lda surilgan joy bekor qilinadi - panel tanlangan burchakka qaytadi
            last_chart_w = -1;
            RelocatePanel();
            ChartRedraw(0);
            return;
           }
        }

      if(shiftDown)
        {
         if(lparam == 49 || lparam == 97)  { robotLang = 1; CreateDashboard(); last_chart_w = -1; RelocatePanel(); UpdateDashboard(); if(newsPanelOpen) { UpdateNewsPanelTexts(); UpdateNewsCountdownLine(); } return; } 
         if(lparam == 50 || lparam == 98)  { robotLang = 2; CreateDashboard(); last_chart_w = -1; RelocatePanel(); UpdateDashboard(); if(newsPanelOpen) { UpdateNewsPanelTexts(); UpdateNewsCountdownLine(); } return; } 
         if(lparam == 51 || lparam == 99)  { robotLang = 3; CreateDashboard(); last_chart_w = -1; RelocatePanel(); UpdateDashboard(); if(newsPanelOpen) { UpdateNewsPanelTexts(); UpdateNewsCountdownLine(); } return; } 

         // --- Shift + X: toggle globalTradeNFP (ACTIVE / BYPASS) ------------------
         if(lparam == 88) { globalTradeNFP = !globalTradeNFP; UpdateDashboard(); ChartRedraw(0); return; }
        }
      else
        {
         bool themeChanged = false;
         
         if(lparam == 48 || lparam == 96)
           {
            rgbMode = !rgbMode;
            if(rgbMode) { fadeProgress = 0.0; currentTargetTheme = currentTheme + 1; if(currentTargetTheme > 9) currentTargetTheme = 1; }
            return;
           }

         if((lparam >= 49 && lparam <= 57) || (lparam >= 97 && lparam <= 105)) rgbMode = false;

         if(lparam == 49 || lparam == 97)  { ApplyTheme(1); themeChanged = true; } 
         if(lparam == 50 || lparam == 98)  { ApplyTheme(2); themeChanged = true; } 
         if(lparam == 51 || lparam == 99)  { ApplyTheme(3); themeChanged = true; } 
         if(lparam == 52 || lparam == 100) { ApplyTheme(4); themeChanged = true; } 
         if(lparam == 53 || lparam == 101) { ApplyTheme(5); themeChanged = true; } 
         if(lparam == 54 || lparam == 102) { ApplyTheme(6); themeChanged = true; } 
         if(lparam == 55 || lparam == 103) { ApplyTheme(7); themeChanged = true; } 
         if(lparam == 56 || lparam == 104) { ApplyTheme(8); themeChanged = true; } 
         if(lparam == 57 || lparam == 105) { ApplyTheme(9); themeChanged = true; } 

         if(themeChanged)
           {
            CreateDashboard();
            last_chart_w = -1;
            RelocatePanel();
            UpdateDashboard();
            if(newsPanelOpen) { UpdateNewsPanelTexts(); UpdateNewsCountdownLine(); }
            return;
           }
        }

      // --- Z va C: Shift bosilmasa ham ishlaydi ---
      if(lparam == 90) { CloseAllPositionsAndOrders(); return; }
      if(lparam == 67) // C -> qo'lda order qo'yish + shu orderlar zonasini chizish/ko'rsatish
        {
         PlaceManualOrders();
         g_zonesVisible = true;
         DrawPendingZones();
         return;
        }

     }
  }

void CreateObjRect(string name, int w, int h, color bg_color, ENUM_BASE_CORNER corner, int zorder)
  {
   ObjectDelete(0, name);
   ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bg_color);
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, name, OBJPROP_COLOR, bg_color);
   ObjectSetInteger(0, name, OBJPROP_CORNER, corner);
   ObjectSetInteger(0, name, OBJPROP_ZORDER, zorder); 
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

void CreateObjText(string name, string text, int size, string font, color clr, ENUM_BASE_CORNER corner, int zorder)
  {
   ObjectDelete(0, name);
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_FONT, font);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, size);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_CORNER, corner);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_ZORDER, zorder); 
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

void CreateObjButton(string name, string text, int w, int h, color bg_color, color txt_color, ENUM_BASE_CORNER corner, int zorder)
  {
   ObjectDelete(0, name);
   ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bg_color);
   ObjectSetInteger(0, name, OBJPROP_COLOR, txt_color);
   ObjectSetInteger(0, name, OBJPROP_BORDER_COLOR, bg_color);
   ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 10);
   ObjectSetInteger(0, name, OBJPROP_CORNER, corner);
   ObjectSetInteger(0, name, OBJPROP_ZORDER, zorder); 
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

//+------------------------------------------------------------------+
//| CreateObjEdit()                                                    |
//| Creates an editable OBJ_EDIT box used for manual lot-size input     |
//| inside the MA'LUMOT (INFO) panel. IMPORTANT: OBJPROP_SELECTABLE     |
//| must stay FALSE - on OBJ_EDIT objects, SELECTABLE=true blocks text  |
//| editing (the object becomes draggable instead), while SELECTABLE=  |
//| false is what actually allows the user to click in and type a new  |
//| lot value. OBJPROP_READONLY is explicitly FALSE so the field stays |
//| user-editable.                                                     |
//+------------------------------------------------------------------+
void CreateObjEdit(string name, string text, int w, int h, color bg_color, color txt_color, ENUM_BASE_CORNER corner, int zorder)
  {
   ObjectDelete(0, name);
   ObjectCreate(0, name, OBJ_EDIT, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bg_color);
   ObjectSetInteger(0, name, OBJPROP_COLOR, txt_color);
   ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 9);
   ObjectSetInteger(0, name, OBJPROP_ALIGN, ALIGN_CENTER);
   ObjectSetInteger(0, name, OBJPROP_CORNER, corner);
   ObjectSetInteger(0, name, OBJPROP_ZORDER, zorder);
   ObjectSetInteger(0, name, OBJPROP_READONLY, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
  }


void CreateDashboard()
  {
   ObjectsDeleteAll(0, prefix);
   ENUM_BASE_CORNER baseCorner = CORNER_LEFT_UPPER;

   string txtBrand = "AZZO TRADE NFP";
   string txtActive = "[ACTIVE]";
   if(robotLang == 3) txtActive = "[АКТИВЕН]";

   if(panelMinimized)
     {
      // --- Kichraytirilgan (minimized) panel: MAIN_PANEL kabi katta emas, faqat
      // --- qisqa "NFP" yorlig'i + TESKARI SANOQ qiymati + ochish("+") tugmasi.
      // --- Qiymatning o'zi UpdateMinimizedCountdown() orqali OnTimer()'da (har
      // --- 50ms'da) yangilanadi - countdownPanelOpen holatidan mustaqil ishlaydi.
      int minW = 175; int minH = 26;
      CreateObjRect(prefix+"min_bg", minW, minH, ColorMainBg, baseCorner, 1);
      CreateObjRect(prefix+"min_border", minW, minH, clrNONE, baseCorner, 2);
      ObjectSetInteger(0, prefix+"min_border", OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, prefix+"min_border", OBJPROP_COLOR, ColorMainNeon);
      CreateObjText(prefix+"brand_txt", "NFP", 8, "Consolas", ColorMainNeon, baseCorner, 3);
      CreateObjText(prefix+"min_cd_txt", "00:00:00", 13, "Consolas", ColorTextPrimary, baseCorner, 3);
      CreateObjRect(prefix+"btn_minimize", 20, 20, ColorChipBg, baseCorner, 3);
      ObjectSetInteger(0, prefix+"btn_minimize", OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, prefix+"btn_minimize", OBJPROP_COLOR, ColorBorderNeon); 
      CreateObjText(prefix+"btn_min_txt", "+", 12, "Consolas", ColorTextPrimary, baseCorner, 4); 
      UpdateMinimizedCountdown(); // darhol to'g'ri qiymat bilan chizamiz ("00:00:00" bir lahza ham ko'rinmasin)
      return;
     }

   CreateObjRect(prefix+"bg", MAIN_PANEL_WIDTH, MAIN_PANEL_HEIGHT, ColorMainBg, baseCorner, 0);         
   CreateObjRect(prefix+"border_neon", MAIN_PANEL_WIDTH, MAIN_PANEL_HEIGHT, clrNONE, baseCorner, 1);
   ObjectSetInteger(0, prefix+"border_neon", OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, prefix+"border_neon", OBJPROP_COLOR, ColorMainNeon); 

   CreateObjRect(prefix+"header", 260, 42, ColorHeaderBg, baseCorner, 2);   

   CreateObjRect(prefix+"btn_minimize", 18, 18, ColorChipBg, baseCorner, 3);
   ObjectSetInteger(0, prefix+"btn_minimize", OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, prefix+"btn_minimize", OBJPROP_COLOR, ColorBorderNeon); 
   CreateObjText(prefix+"btn_min_txt", "-", 12, "Consolas", ColorTextPrimary, baseCorner, 4); 

   CreateObjText(prefix+"title_text", ">BISMILLAH<", 13, "Consolas", ColorMainNeon, baseCorner, 4); 
   CreateObjText(prefix+"status_txt", txtActive, 9, "Consolas", C'0,255,102', baseCorner, 4); 

   string lblSym="SYMBOL     :", lblBal="BALANCE    :", lblSpd="SPREAD     :", lblSlp="MAX SLIP   :";
   string lblLev="LEVERAGE   :", lblLoc="LOCAL TIME :", lblSrv="SERVER TIME:", lblTrig="TARGET NFP :";
   string lblWin="TIME LIMIT :", lblPing="BROKER PING:";
   string lblDir="DIRECTION  :", lblMaxLot="MAX LOT    :";
   string bSet="[ ORDER QO'YISH ]", bClose="[ HAMMASINI TOZALASH ]";
   string bInfo = infoPanelOpen ? "[ CLOSE ]" : "[ INFO (I) ]";
   string bNews = newsPanelOpen ? "[ CLOSE ]" : "F.FACTORY [F]";
   string bCd   = countdownPanelOpen ? "[ CLOSE COUNTDOWN ]" : "Countdown [ T ]";
   string bRetr = retracementPanelOpen ? "[ CLOSE ]" : "RETRACEMENT [R]";

   // --- YANGI: INFO paneldagi "orderlar orasidagi masofa" (Grid Step) qatorlari
   // --- uchun sarlavha va uchta label (1-5 orasi / 6-50 orasi / 2 guruh orasi).
   string lblGridTitle = "ORDERLAR ORASI";
   string lblGridStep1 = "1-5         :";
   string lblGridStep2 = "6-50        :";
   string lblGridGap   = "2 GURUH ORASI:";
   string lblGridDist  = "MASOFASI    :";
   string lblGridSL    = "SL          :";
   
   string currentFont = "Consolas";

   if(robotLang == 1) // UZB
     {
      lblSym="PRODUKT    :"; lblBal="BALANS     :"; lblSpd="SPREAD     :"; lblSlp="MAKS SLIP  :";
      lblLev="KREDIT LEL :"; lblLoc="MAHALLIY VR:"; lblSrv="SERVER VR  :"; lblTrig="NFP VAQTI  :";
      lblWin="LIMIT VAQT :"; lblPing="BROKER PING:";
      lblDir="YO'NALISH  :"; lblMaxLot="MAKS LOT   :";
      bSet="[ ORDER QO'YISH ]"; bClose="[ HAMMASINI TOZALASH ]";
      bInfo = infoPanelOpen ? "[ YOPISH ]" : "[ INFO (I) ]";
      bNews = newsPanelOpen ? "[ YOPISH ]" : "F.FACTORY [F]";
      bCd   = countdownPanelOpen ? "[ SANOQNI YOPISH ]" : "Teskari Sanoq [ T ]";
      bRetr = retracementPanelOpen ? "[ YOPISH ]" : "RETRACEMENT [R]";
      lblGridTitle = "ORDERLAR ORASI";
      lblGridStep1 = "1-5         :";
      lblGridStep2 = "6-50        :";
      lblGridGap   = "2 GURUH ORASI:";
      lblGridDist  = "MASOFASI    :";
      lblGridSL    = "SL          :";
     }
   else if(robotLang == 3) // RUS
     {
      currentFont = "Arial";
      lblSym="ИНСТРУМЕНТ :"; lblBal="БАЛАНС     :"; lblSpd="СПРЕД      :"; lblSlp="МАКС ПРОСКА:";
      lblLev="ПЛЕЧО      :"; lblLoc="МЕСТНОЕ ВР :"; lblSrv="ВР СЕРВЕРА :"; lblTrig="ВРЕМЯ NFP  :";
      lblWin="ТАЙМЛИМИТ  :"; lblPing="ПИНГ СЕРВЕРА:";
      lblDir="НАПРАВЛЕНИЕ:"; lblMaxLot="МАКС ЛОТ   :";
      bSet="[ ВЫСТАВИТЬ ОРДЕРА ]"; bClose="[ ЗАКРЫТЬ ВСЕ ]";
      bInfo = infoPanelOpen ? "[ ЗАКРЫТЬ ]" : "[ ИНФО (I) ]";
      bNews = newsPanelOpen ? "[ ЗАКРЫТЬ ]" : "F.FACTORY [F]";
      bCd   = countdownPanelOpen ? "[ ЗАКРЫТЬ ОТСЧЁТ ]" : "Обратный отсчёт [ T ]";
      bRetr = retracementPanelOpen ? "[ ЗАКРЫТЬ ]" : "RETRACEMENT [R]";
      lblGridTitle = "ШАГ МЕЖДУ ОРДЕРАМИ";
      lblGridStep1 = "1-5          :";
      lblGridStep2 = "6-50         :";
      lblGridGap   = "МЕЖДУ ГРУПП. :";
      lblGridDist  = "ДИСТАНЦИЯ    :";
      lblGridSL    = "SL           :";
     }

   CreateObjText(prefix+"lbl_sym", lblSym, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_sym", _Symbol, 9, "Consolas", ColorMainNeon, baseCorner, 4);
   
   CreateObjText(prefix+"lbl_bal", lblBal, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_bal", "0.00", 9, "Consolas", ColorMainNeon, baseCorner, 4); 
   
   CreateObjText(prefix+"lbl_spd", lblSpd, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_spd", "0", 9, "Consolas", ColorMainNeon, baseCorner, 4); 
   
   CreateObjText(prefix+"lbl_slp", lblSlp, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_slp", (string)InpMaxSlippage + " Pts", 9, "Consolas", ColorMainNeon, baseCorner, 4);
   
   CreateObjText(prefix+"lbl_lev", lblLev, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_lev", "1:0", 9, "Consolas", ColorMainNeon, baseCorner, 4);
   
   CreateObjText(prefix+"lbl_loc", lblLoc, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_loc", "00:00:00", 10, "Consolas", C'210,215,220', baseCorner, 4); 

   CreateObjText(prefix+"lbl_srv", lblSrv, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_srv", "00:00:00", 10, "Consolas", clrYellow, baseCorner, 4); 
   
   CreateObjText(prefix+"lbl_trig", lblTrig, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   // --- YANGI: NFP vaqti endi qayta compile qilmasdan, to'g'ridan-to'g'ri panel
   // ustida (val_trig edit maydonida) o'zgartirilishi mumkin. Kiritilgan qiymat
   // CHARTEVENT_OBJECT_ENDEDIT hodisasida (pastda) tekshiriladi:
   //  - format xato bo'lsa (masalan harf/belgi) -> eski qiymatga qaytadi
   //  - 00:00:00-23:59:59 diapazonidan tashqari bo'lsa -> "15:29:55" ga qaytadi
   CreateObjEdit(prefix+"val_trig", g_nfpTime, 68, 18, C'25,25,25', clrRed, baseCorner, 4);
   
   CreateObjText(prefix+"lbl_win", lblWin, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   string suffWin = (robotLang == 3) ? " Сек" : " Sek";
   CreateObjText(prefix+"val_win", (string)InpTimeWindowSeconds + suffWin, 9, "Consolas", ColorMainNeon, baseCorner, 4);

   CreateObjText(prefix+"lbl_ping", lblPing, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_ping", "0 ms", 9, "Consolas", ColorMainNeon, baseCorner, 4);

   CreateObjButton(prefix+"btn_set", bSet, 240, 26, C'20,35,35', ColorMainNeon, baseCorner, 5);
   CreateObjButton(prefix+"btn_closeall", bClose, 240, 26, C'45,15,15', clrRed, baseCorner, 5);
   
   CreateObjButton(prefix+"btn_info", bInfo, 118, 26, C'45,15,15', clrRed, baseCorner, 5);
   // --- F.FACTORY tugmasi endi INFO tugmasi bilan BIR XIL qizil uslubda ---
   // --- F.FACTORY tugmasi endi INFO tugmasi bilan TO'LIQ bir xil: chegara
   // --- rangi alohida o'rnatilmaydi (default bo'yicha fon rangi bilan bir xil
   // --- bo'lib, atrofida ortiqcha chiziq/ramka ko'rinmaydi) ---
   CreateObjButton(prefix+"btn_newstoggle", bNews, 118, 26, C'45,15,15', clrRed, baseCorner, 5);
   ObjectSetInteger(0, prefix+"btn_newstoggle", OBJPROP_FONTSIZE, 9); // "F.FACTORY [F]" 118px ichiga to'liq sig'ishi uchun standart 10dan kichikroq
   CreateObjButton(prefix+"btn_cd_toggle", bCd, 240, 26, C'20,35,35', ColorMainNeon, baseCorner, 5);
   // --- YANGI: "RETRACEMENT [R]" tugmasi - "Teskari Sanoq" tugmasi bilan BIR XIL
   // --- ustunda, uning PASTIDA joylashadi (tepada RETRACEMENT, pastda Teskari Sanoq) ---
   CreateObjButton(prefix+"btn_retracement", bRetr, 240, 26, C'20,35,35', ColorMainNeon, baseCorner, 5);
   ObjectSetInteger(0, prefix+"btn_retracement", OBJPROP_FONTSIZE, 9);

   // --- AZZO ML: "YO'NALISH:" qatori - endi tepadagi boshqa qatorlar (masalan
   // --- BALANCE) bilan AYNAN bir xil uslubda: label ColorBorderNeon/currentFont,
   // --- qiymat ColorMainNeon/Consolas. Faqat vizual - savdo logikasiga ta'sir qilmaydi.
   CreateObjText(prefix+"lbl_dir", lblDir, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_dir", g_dirText, 9, "Consolas", g_dirColor, baseCorner, 4);

   // --- YANGI: "MAKSIMAL LOT:" qatori - joriy balans/equity, leverage va
   // --- instrumentning narxiga qarab ochish mumkin bo'lgan MAKSIMAL lot
   // --- hajmini hisoblab ko'rsatadi (barcha juftliklar uchun ishlaydi,
   // --- OrderCalcMargin() orqali). Faqat vizual - savdo logikasiga ta'sir qilmaydi.
   CreateObjText(prefix+"lbl_maxlot", lblMaxLot, 9, currentFont, ColorBorderNeon, baseCorner, 4);
   CreateObjText(prefix+"val_maxlot", "0.00", 9, "Consolas", ColorTextPrimary, baseCorner, 4);

   if(robotLang == 3)
     {
      ObjectSetString(0, prefix+"btn_set", OBJPROP_FONT, "Arial");
      ObjectSetString(0, prefix+"btn_closeall", OBJPROP_FONT, "Arial");
      ObjectSetString(0, prefix+"btn_info", OBJPROP_FONT, "Arial");
      ObjectSetString(0, prefix+"btn_newstoggle", OBJPROP_FONT, "Arial");
      ObjectSetString(0, prefix+"btn_cd_toggle", OBJPROP_FONT, "Arial");
      ObjectSetString(0, prefix+"btn_retracement", OBJPROP_FONT, "Arial");
     }

   // --- Visual Button Hover FX: tugmalar hozirgina qayta yaratilgani uchun
   // --- (bazaviy ranglarga qaytgan) hover ro'yxati ham shu yerda qayta
   // --- ro'yxatga olinadi - shunda eski "yopishib qolgan" hover holati
   // --- panel qayta chizilganda saqlanib qolmaydi.
   InitHoverButtons();

   // --- INFO PANEL ---
   if(infoPanelOpen)
     {
      int activeDisplayRows = g_orderCount;
      if (activeDisplayRows < 1)  activeDisplayRows = 1;
      if (activeDisplayRows > AZZO_MAX_GRID_ORDERS) activeDisplayRows = AZZO_MAX_GRID_ORDERS;

      // Panel balandligi endi chap tarafdagi asosiy "AZZO NEON PRO" paneli bilan
      // AYNAN BIR XIL (MAIN_PANEL_HEIGHT) qilib belgilanadi - hech qanday kalta yoki
      // nomutanosib (qisqa) INFO paneli qolmasligi uchun. Ikkala panel ham bir xil
      // YDISTANCE (py) dan boshlanadi (RelocatePanel() da), shuning uchun ular endi
      // to'liq teng va parallel ko'rinadi.
      int infoHeight = MAIN_PANEL_HEIGHT;
      int infoWidth  = 215; // 180 -> 215: yangi "orderlar orasi" qatorlari va uzunroq label'lar sig'ishi uchun kattalashtirildi

      CreateObjRect(prefix+"info_bg", infoWidth, infoHeight, ColorMainBg, baseCorner, 0);
      CreateObjRect(prefix+"info_border", infoWidth, infoHeight, clrNONE, baseCorner, 1);
      ObjectSetInteger(0, prefix+"info_border", OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, prefix+"info_border", OBJPROP_COLOR, ColorMainNeon);

      CreateObjRect(prefix+"info_header", infoWidth, 42, ColorHeaderBg, baseCorner, 2);
      string infoTitle = (robotLang == 1) ? "MA'LUMOT" : ((robotLang == 3) ? "ИНФО" : "INFORMATION");
      CreateObjText(prefix+"info_title", infoTitle, 10, "Consolas", ColorMainNeon, baseCorner, 3);

      // Har bir qator: nom labeli (masalan "Order 3 :" yoki guruhlangan "6-20 :")
      // + tahrirlanadigan (Edit) lot maydoni + "Lot" birlik labeli.
      // Birinchi 5 order HAR DOIM alohida ko'rsatiladi (1,2,3,4,5), qolgan
      // BARCHA orderlar esa BITTA yagona guruh qatoriga tushadi (masalan
      // orderCount=20 bo'lsa - "6-20"; orderCount=50 bo'lsa - "6-50") -
      // BuildInfoRows() shu diapazonni g_orderCount asosida hisoblab beradi.
      {
       int rowStart[]; int rowEnd[];
       int rowCount = BuildInfoRows(activeDisplayRows, rowStart, rowEnd);

       for(int i = 0; i < rowCount; i++)
         {
          string lvlName = (robotLang == 3) ? "Ордер " : "Order ";
          string rowLabel;
          if(rowStart[i] == rowEnd[i])
             rowLabel = lvlName + (string)(rowStart[i]+1) + " :";
          else
             rowLabel = (string)(rowStart[i]+1) + "-" + (string)(rowEnd[i]+1) + " :";

          CreateObjText(prefix+"info_lbl_"+(string)i, rowLabel, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
          CreateObjEdit(prefix+"info_val_"+(string)i, DoubleToString(inputLots[rowStart[i]], 2), 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);
          CreateObjText(prefix+"info_unit_"+(string)i, "Lot", 9, "Consolas", ColorBorderNeon, baseCorner, 4);
         }
      }

      // --- YANGI: "Orderlar soni" maydoni -----------------------------------------
      // Bu yerga 1 dan AZZO_MAX_GRID_ORDERS gacha son kiritilsa, PlaceStraddleOrders()
      // aynan shuncha ta BUY STOP va shuncha ta SELL STOP qo'yadi (masalan 1 -> 1
      // Buy Stop + 1 Sell Stop, 10 -> 10 Buy Stop + 10 Sell Stop). Qiymat undan
      // katta bo'lishi mumkin emas - avtomatik ravishda chegaraga cheklanadi
      // (ENDEDIT hodisasida).
      string ocLbl = (robotLang == 1) ? "ORDERLAR SONI (1-"+(string)AZZO_MAX_GRID_ORDERS+"):" : ((robotLang == 3) ? "КОЛ-ВО ОРДЕРОВ(1-"+(string)AZZO_MAX_GRID_ORDERS+"):" : "ORDER COUNT (1-"+(string)AZZO_MAX_GRID_ORDERS+") :");
      CreateObjText(prefix+"info_oc_lbl", ocLbl, 8, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"info_ordercount", (string)g_orderCount, 50, 20, C'25,25,25', ColorMainNeon, baseCorner, 4);

      // --- YANGI: "ORDERLAR ORASI" (Grid Step) bloki - 1-5 orderlar orasidagi
      // --- masofa, 6-50 orderlar orasidagi masofa va 2 guruh orasidagi qo'shimcha
      // --- masofa endi to'g'ridan-to'g'ri panel ustida (qayta compile qilmasdan)
      // --- tahrirlanadi. Tepadagi "Order N :" Lot qatorlari bilan AYNAN BIR XIL
      // --- uslub: label (ColorBorderNeon) + Edit maydon (ColorMainNeon fon) +
      // --- "PT" (Punkt) birlik labeli.
      CreateObjText(prefix+"info_grid_title", lblGridTitle, 8, "Consolas", ColorMainNeon, baseCorner, 4);

      CreateObjText(prefix+"info_grid_lbl_1", lblGridStep1, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"info_grid_val_1", (string)g_gridStep1, 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);
      CreateObjText(prefix+"info_grid_unit_1", "PT", 9, "Consolas", ColorBorderNeon, baseCorner, 4);

      CreateObjText(prefix+"info_grid_lbl_2", lblGridStep2, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"info_grid_val_2", (string)g_gridStep2, 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);
      CreateObjText(prefix+"info_grid_unit_2", "PT", 9, "Consolas", ColorBorderNeon, baseCorner, 4);

      CreateObjText(prefix+"info_grid_lbl_3", lblGridGap, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"info_grid_val_3", (string)g_gridGap, 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);
      CreateObjText(prefix+"info_grid_unit_3", "PT", 9, "Consolas", ColorBorderNeon, baseCorner, 4);

      // --- YANGI: "MASOFASI" (order narxining Ask/Bid'dan uzoqligi) va "SL"
      // --- (Stop Loss masofasi) maydonlari - "2 GURUH ORASI" qatoridan PASTDA,
      // --- AYNAN BIR XIL uslubda. Default ikkalasi ham 750 PUNKT (= 7.5$) -
      // --- InpDistance / InpStopLoss qiymatlaridan (OnInit()da) boshlang'ich holatga
      // --- keltiriladi, keyin esa FAQAT shu maydonlar orqali (qayta compile
      // --- qilmasdan) o'zgartiriladi.
      CreateObjText(prefix+"info_grid_lbl_4", lblGridDist, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"info_grid_val_4", (string)g_orderDistance, 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);
      CreateObjText(prefix+"info_grid_unit_4", "PT", 9, "Consolas", ColorBorderNeon, baseCorner, 4);

      CreateObjText(prefix+"info_grid_lbl_5", lblGridSL, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"info_grid_val_5", (string)g_orderSL, 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);
      CreateObjText(prefix+"info_grid_unit_5", "PT", 9, "Consolas", ColorBorderNeon, baseCorner, 4);

      // --- YANGI: TAYYOR SOZLAMALAR (PRESET) TUGMALARI - "NFP", "CPI" va "OTHER NEWS" ---
      // Bosilganda AZZO_ApplyPreset*() (yuqorida, HandlePanelButtonClick() ichida chaqiriladi)
      // orqali BARCHA yuqoridagi maydonlar (Orderlar soni, Lot qatorlari, Orderlar orasi,
      // Masofasi, SL) bitta bosishda oldindan belgilangan qiymatlarga o'rnatiladi.
      // "NFP" va "CPI" YONMA-YON (bir qatorda), "OTHER NEWS" esa ularning PASTIDA,
      // to'liq kenglikda joylashadi - xuddi rasmdagi maketga mos.
      string presetTitle = (robotLang == 1) ? "TAYYOR SOZLAMALAR" : ((robotLang == 3) ? "ГОТОВЫЕ НАСТРОЙКИ" : "QUICK PRESETS");
      CreateObjText(prefix+"info_preset_title", presetTitle, 8, "Consolas", ColorMainNeon, baseCorner, 4);

      CreateObjButton(prefix+"info_btn_nfp", "NFP", 90, 24, C'20,35,35', ColorMainNeon, baseCorner, 5);
      CreateObjButton(prefix+"info_btn_cpi", "CPI", 90, 24, C'20,35,35', ColorMainNeon, baseCorner, 5);
      CreateObjButton(prefix+"info_btn_othernews", "OTHER NEWS", 185, 24, C'20,35,35', ColorMainNeon, baseCorner, 5);
      ObjectSetInteger(0, prefix+"info_btn_nfp", OBJPROP_FONTSIZE, 9);
      ObjectSetInteger(0, prefix+"info_btn_cpi", OBJPROP_FONTSIZE, 9);
      ObjectSetInteger(0, prefix+"info_btn_othernews", OBJPROP_FONTSIZE, 9);
     }

   // --- RETRACEMENT (LIMIT ORDERS) PANELI ---
   // MA'LUMOT (INFO) panelining ANIQ NUSXASI - endi FAQAT fon/chegara/sarlavha
   // uslubi emas, balki O'LCHAMI (eni 215px, INFO bilan bir xil), qator uslubi
   // (label + Edit maydon BIR QATORDA, xuddi "ORDERLAR ORASI" qatorlari kabi)
   // va RANGLARI (label - ColorBorderNeon, edit matni - ColorMainNeon, edit foni
   // C'25,25,25') ham INFO panel bilan TO'LIQ BIR XIL qilib qayta ishlandi.
   // Sarlavhada "2-ETAB" o'rniga endi "LIMIT ORDERS" yoziladi, tepasida "HOLAT :"
   // yoki "RETRACEMENT [R]" tugmasi bilan mustaqil ochiladi/yopiladi, va
   // asosiy panel bilan (RelocatePanel() orqali) TO'LIQ SINXRON harakatlanadi.
   if(retracementPanelOpen)
     {
      int retrHeight = MAIN_PANEL_HEIGHT; // asosiy panel / INFO bilan bir xil balandlik
      int retrWidth  = 215;               // INFO panel bilan bir xil eni

      CreateObjRect(prefix+"retr_bg", retrWidth, retrHeight, ColorMainBg, baseCorner, 0);
      CreateObjRect(prefix+"retr_border", retrWidth, retrHeight, clrNONE, baseCorner, 1);
      ObjectSetInteger(0, prefix+"retr_border", OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, prefix+"retr_border", OBJPROP_COLOR, ColorMainNeon);

      CreateObjRect(prefix+"retr_header", retrWidth, 42, ColorHeaderBg, baseCorner, 2);
      // --- "LIMIT ORDERS" - MA'LUMOT panelidagi sarlavha bilan BIR XIL shrift o'lchamida (10pt) ---
      CreateObjText(prefix+"retr_title", "LIMIT ORDERS", 10, "Consolas", ColorMainNeon, baseCorner, 3);

      // --- YANGI: "HOLAT" (Status) tugmasi - g_recBlockStopOrders true/false
      // --- ekanini panelning ENG TEPASIDA (Trigger'dan OLDIN) ko'rsatadi VA
      // --- bosilganda TRUE <-> FALSE almashadi (HandlePanelButtonClick() da).
      // --- TRUE bo'lsa yashil, FALSE bo'lsa qizil fonda/matnda chiqadi.
      string lblHolat = (robotLang == 3) ? "СТАТУС       :" : "HOLAT        :";
      string valHolat = g_recBlockStopOrders ? "TRUE" : "FALSE";
      color  colHolat = g_recBlockStopOrders ? C'0,255,102' : clrRed;
      CreateObjText(prefix+"retr_lbl_status", lblHolat, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjButton(prefix+"retr_val_status", valHolat, 55, 18, C'25,25,25', colHolat, baseCorner, 4);
      ObjectSetInteger(0, prefix+"retr_val_status", OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, prefix+"retr_val_status", OBJPROP_BORDER_COLOR, colHolat);

      string rLbl0 = (robotLang == 3) ? "ТРИГГЕР (ПУНКТ):"    : "TRIGGER (PUNKT)  :";
      string rLbl1 = (robotLang == 3) ? "1-УРОВЕНЬ (ПУНКТ):"  : "1-DARAJA (PUNKT) :";
      string rLbl2 = (robotLang == 3) ? "ШАГ (ПУНКТ):"        : "QADAM (PUNKT)    :";
      string rLbl3 = (robotLang == 3) ? "ОРДЕРОВ (ШТ):"       : "ORDERLAR SONI    :";
      string rLbl4 = (robotLang == 3) ? "ЛОТ:"                : "LOT              :";

      // Har bir qator: label (ColorBorderNeon, 9pt) + tahrirlanadigan Edit maydon
      // (ColorMainNeon matn, C'25,25,25' fon) - AYNAN INFO paneldagi "ORDERLAR
      // ORASI" qatorlari bilan bir xil uslub va rang. Rows 1-4 - butun son (INT),
      // Row 5 (Lot) - kasr son (DOUBLE, 2 xonagacha).
      CreateObjText(prefix+"retr_lbl_0", rLbl0, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"retr_val_0", (string)g_recTrigger, 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);

      CreateObjText(prefix+"retr_lbl_1", rLbl1, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"retr_val_1", (string)g_recOffset, 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);

      CreateObjText(prefix+"retr_lbl_2", rLbl2, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"retr_val_2", (string)g_recStep, 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);

      CreateObjText(prefix+"retr_lbl_3", rLbl3, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"retr_val_3", (string)g_recLevels, 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);

      CreateObjText(prefix+"retr_lbl_4", rLbl4, 9, "Consolas", ColorBorderNeon, baseCorner, 4);
      CreateObjEdit(prefix+"retr_val_4", DoubleToString(g_recLot, 2), 55, 18, C'25,25,25', ColorMainNeon, baseCorner, 4);

      if(robotLang == 3) ObjectSetString(0, prefix+"retr_title", OBJPROP_FONT, "Arial");
     }

   // --- TESKARI SANOQ (COUNTDOWN) PANELI ---
   // INFO panelidan PASTDA joylashadi (aniq X/Y RelocatePanel()'da hisoblanadi).
   // "T" tugmasi bilan mustaqil ochiladi/yopiladi - INFO ochiq yoki yopiqligidan
   // qat'i nazar ishlaydi.
   if(countdownPanelOpen)
     {
      int cdWidth  = MAIN_PANEL_WIDTH; // asosiy panel bilan bir xil kenglik (260)
      int cdHeight = 150;              // YANA HAM KATTALASHTIRILDI (130 -> 150)

      CreateObjRect(prefix+"cd_bg", cdWidth, cdHeight, ColorMainBg, baseCorner, 0);
      CreateObjRect(prefix+"cd_border", cdWidth, cdHeight, clrNONE, baseCorner, 1);
      ObjectSetInteger(0, prefix+"cd_border", OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, prefix+"cd_border", OBJPROP_COLOR, ColorMainNeon);

      CreateObjRect(prefix+"cd_header", cdWidth, 26, ColorHeaderBg, baseCorner, 2);
      string cdTitle = (robotLang == 1) ? "TESKARI SANOQ" : ((robotLang == 3) ? "ОБРАТНЫЙ ОТСЧЁТ" : "COUNTDOWN");
      CreateObjText(prefix+"cd_title", cdTitle, 10, "Consolas", ColorMainNeon, baseCorner, 3);

      // Boshlang'ich matn - haqiqiy qiymat UpdateCountdownPanel() tomonidan
      // (OnTimer ichida har 50ms'da) darhol yangilanadi. ANCHOR_CENTER orqali
      // matn uzunligidan qat'i nazar (masalan "59" yoki "1:00:10") panel
      // ichida HAM GORIZONTAL, HAM VERTIKAL markazda turadi.
      CreateObjText(prefix+"cd_value", "00:00:00", 36, "Consolas", ColorTextPrimary, baseCorner, 4);
      ObjectSetInteger(0, prefix+"cd_value", OBJPROP_ANCHOR, ANCHOR_CENTER);
     }

   // --- YANGILIKLAR KALENDARI (F.FACTORY) PANELI ---
   // ForexFactory.com saytining O'ZIGA o'xshab chiziladi: to'q ko'k (navy) sarlavha,
   // chapda saytdagi qizil "High Impact" papka belgisiga o'xshash qizil chiziq,
   // valyuta uchun alohida rangli "badge" va navbatma-navbat (zebra) qator foni.
   // Ranglar FF_* konstantalaridan olinadi - asosiy panel NEON mavzusidan mustaqil,
   // shu tufayli F.FACTORY paneli har doim "sayt" ko'rinishida qoladi.
   // "F.FACTORY [ F ]" tugmasi yoki "F" tez tugmasi bilan mustaqil ochiladi/yopiladi.
   if(newsPanelOpen)
     {
      int newsRows = InpNewsCalendarCount;
      if(newsRows < 1) newsRows = 1;
      if(newsRows > NEWS_MAX_ROWS) newsRows = NEWS_MAX_ROWS;

      int newsWidth   = 320;
      int newsHeaderH = 30;
      int newsCdRowH  = 28;
      int newsRowH    = 40;
      int newsHeight  = newsHeaderH + newsCdRowH + (newsRows * newsRowH) + 10;

      CreateObjRect(prefix+"news_bg", newsWidth, newsHeight, FF_BG, baseCorner, 0);
      CreateObjRect(prefix+"news_border", newsWidth, newsHeight, clrNONE, baseCorner, 1);
      ObjectSetInteger(0, prefix+"news_border", OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, prefix+"news_border", OBJPROP_COLOR, ColorMainNeon); // INFO panelidagi bilan bir xil chegara rangi

      // --- Sarlavha: sayt navigatsiya panelidagi kabi to'q ko'k, INFO panelidagi
      // --- kabi qizil chekka chiziqsiz (neytral) uslubda ---
      CreateObjRect(prefix+"news_header", newsWidth, newsHeaderH, FF_HEADER, baseCorner, 2);
      string newsTitle = "FOREX FACTORY   (TOP-" + (string)newsRows + ")";
      CreateObjText(prefix+"news_title", newsTitle, 11, "Consolas", ColorMainNeon, baseCorner, 3);

      // --- "Keyingisigacha" (Up Next) qatori - saytdagi "Up Next" belgisiga o'xshab qizil qiymat ---
      // Endi kattaroq (12->14) va yanada "bomba" ko'rinadigan hisoblagich.
      string newsCdLbl = (robotLang == 1) ? "Keyingisigacha:" : ((robotLang == 3) ? "До следующей:" : "Next in:");
      CreateObjRect(prefix+"news_cd_bg", newsWidth, newsCdRowH, FF_HEADER, baseCorner, 2);
      CreateObjText(prefix+"news_cd_lbl", newsCdLbl, 9, "Consolas", FF_SUBTEXT, baseCorner, 4);
      CreateObjText(prefix+"news_cd_val", "--:--:--", 14, "Consolas", FF_RED, baseCorner, 4);

      // --- Har bir qator: zebra fon + qizil "impact" chizig'i + valyuta "badge"i
      // --- (endi ASOSIY panel neon rangida chegaralangan - sinxron ko'rinish) +
      // --- sana/kun/vaqt (CYAN rangda, kattaroq va aniq ko'rinadigan shriftda - qizil EMAS) +
      // --- yangilik NOMI (katta, oq - asosiy diqqat, o'zgarishsiz) ---
      for(int i = 0; i < newsRows; i++)
        {
         color stripeClr = (i % 2 == 0) ? FF_STRIPE_A : FF_STRIPE_B;
         CreateObjRect(prefix+"news_stripe_"+(string)i, newsWidth, newsRowH, stripeClr, baseCorner, 1);
         CreateObjRect(prefix+"news_impact_"+(string)i, 5, newsRowH - 8, FF_BG, baseCorner, 2);
         CreateObjRect(prefix+"news_curbg_"+(string)i, 38, 16, FF_BG, baseCorner, 2);
         ObjectSetInteger(0, prefix+"news_curbg_"+(string)i, OBJPROP_COLOR, ColorMainNeon); // badge chegarasi - ASOSIY panel bilan sinxron
         CreateObjText(prefix+"news_cur_"+(string)i, "", 8, "Consolas", ColorTextPrimary, baseCorner, 4);
         CreateObjText(prefix+"news_row1_"+(string)i, "", 9, "Consolas", FF_CYAN, baseCorner, 4);
         CreateObjText(prefix+"news_row2_"+(string)i, "Yuklanmoqda...", 11, "Consolas", ColorTextPrimary, baseCorner, 4);
         ObjectSetInteger(0, prefix+"news_curbg_"+(string)i, OBJPROP_ZORDER, 2);
        }
     }

  }

void RelocatePanel(bool force = false)
  {
   int chart_w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int chart_h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
   if(chart_w < 100 || chart_h < 100) return;
   if(!force && chart_w == last_chart_w && chart_h == last_chart_h && currentCorner == last_corner) return;
      
   last_chart_w = chart_w; last_chart_h = chart_h; last_corner = currentCorner;
   int px = InpOffsetX; int py = InpOffsetY;

   if(panelMinimized)
     {
      int b_h = 26; int b_w = 175; 
      if(g_manualPos)
        {
         px = g_panelX; py = g_panelY;
        }
      else if(currentCorner == Corner_Right_Upper) { px = chart_w - b_w - InpOffsetX; py = InpOffsetY; }
      else if(currentCorner == Corner_Left_Lower)  { px = InpOffsetX; py = chart_h - b_h - InpOffsetY; }
      else if(currentCorner == Corner_Right_Lower) { px = chart_w - b_w - InpOffsetX; py = chart_h - b_h - InpOffsetY; }

      // --- Panel chart chegarasidan chiqib ketmasligi uchun cheklov (clamp) ---
      if(px < 0) px = 0;
      if(py < 0) py = 0;
      if(px > chart_w - b_w) px = chart_w - b_w;
      if(py > chart_h - b_h) py = chart_h - b_h;
      if(g_manualPos) { g_panelX = px; g_panelY = py; }
      
      ObjSetXY(prefix+"min_bg", px, py);
      ObjSetXY(prefix+"min_border", px, py);
      ObjSetXY(prefix+"brand_txt", px + 8, py + 8);
      ObjSetXY(prefix+"min_cd_txt", px + 34, py + 5);
      ObjSetXY(prefix+"btn_minimize", px + 151, py + 3);
      ObjSetXY(prefix+"btn_min_txt", px + 156, py + 3);
      return;
     }

   int p_w = MAIN_PANEL_WIDTH; int p_h = MAIN_PANEL_HEIGHT; 
   if(g_manualPos)
     {
      px = g_panelX; py = g_panelY;
     }
   else if(currentCorner == Corner_Right_Upper) { px = chart_w - p_w - InpOffsetX; py = InpOffsetY; }
   else if(currentCorner == Corner_Left_Lower)  { px = InpOffsetX; py = chart_h - p_h - InpOffsetY; }
   else if(currentCorner == Corner_Right_Lower) { px = chart_w - p_w - InpOffsetX; py = chart_h - p_h - InpOffsetY; }
   if(px < 0) px = 0; if(py < 0) py = 0; 

   // --- Panel chart chegarasidan (o'ng/pastki tomondan) chiqib ketmasligi uchun
   // --- cheklov. Yon panellar (INFO/F.FACTORY) ochiq bo'lsa ham asosiy panel
   // --- doim chart ichida qoladi; faqat asosiy panel kengligi hisobga olinadi,
   // --- chunki yon panellar undan o'ngga qarab qo'shimcha chiziladi.
   if(px > chart_w - p_w) px = chart_w - p_w;
   if(py > chart_h - p_h) py = chart_h - p_h;
   if(px < 0) px = 0; if(py < 0) py = 0;
   if(g_manualPos) { g_panelX = px; g_panelY = py; }

   ObjSetXY(prefix+"bg", px, py);
   ObjSetXY(prefix+"border_neon", px, py);
   ObjSetXY(prefix+"header", px, py);
   
   ObjSetXY(prefix+"btn_minimize", px+10, py+12);
   ObjSetXY(prefix+"btn_min_txt", px+15, py+12);
   ObjSetXY(prefix+"title_text", px+36, py+14);
   ObjSetXY(prefix+"status_txt", px+185, py+14); 
   

   int startY = py + 52; int stepY = 21; 
   ObjSetXY(prefix+"lbl_sym", px+15, startY);              ObjSetXY(prefix+"val_sym", px+135, startY);
   ObjSetXY(prefix+"lbl_bal", px+15, startY + stepY);      ObjSetXY(prefix+"val_bal", px+135, startY + stepY);
   ObjSetXY(prefix+"lbl_spd", px+15, startY + (stepY*2));  ObjSetXY(prefix+"val_spd", px+135, startY + (stepY*2));
   ObjSetXY(prefix+"lbl_slp", px+15, startY + (stepY*3));  ObjSetXY(prefix+"val_slp", px+135, startY + (stepY*3));
   ObjSetXY(prefix+"lbl_lev", px+15, startY + (stepY*4));  ObjSetXY(prefix+"val_lev", px+135, startY + (stepY*4));
   
   ObjSetXY(prefix+"lbl_loc", px+15, startY + (stepY*5));  ObjSetXY(prefix+"val_loc", px+135, startY + (stepY*5));
   ObjSetXY(prefix+"lbl_srv", px+15, startY + (stepY*6));  ObjSetXY(prefix+"val_srv", px+135, startY + (stepY*6));
   ObjSetXY(prefix+"lbl_trig", px+15, startY + (stepY*7)); ObjSetXY(prefix+"val_trig", px+135, startY + (stepY*7));
   ObjSetXY(prefix+"lbl_win", px+15, startY + (stepY*8));  ObjSetXY(prefix+"val_win", px+135, startY + (stepY*8));
   ObjSetXY(prefix+"lbl_ping", px+15, startY + (stepY*9));  ObjSetXY(prefix+"val_ping", px+135, startY + (stepY*9));

   // --- AZZO ML: "YO'NALISH:" va "MAKSIMAL LOT:" qatorlari - barcha ma'lumot
   // --- yozuvlaridan (lbl/val qatorlaridan) KEYIN, lekin tugmalardan OLDIN
   // --- joylashtirildi (xuddi shu 21px panjara bo'yicha, tepadagi qatorlar kabi) ---
   ObjSetXY(prefix+"lbl_dir", px+15, py + 262);     ObjSetXY(prefix+"val_dir", px+135, py + 262);
   ObjSetXY(prefix+"lbl_maxlot", px+15, py + 283);  ObjSetXY(prefix+"val_maxlot", px+135, py + 283);

   ObjSetXY(prefix+"btn_set", px+10, py + 322); 
   ObjSetXY(prefix+"btn_closeall", px+10, py + 354); 
   
   ObjSetXY(prefix+"btn_info", px+10, py + 386); 
   ObjSetXY(prefix+"btn_newstoggle", px+132, py + 386); 

   // --- YANGI: "RETRACEMENT [ R ]" tugmasi TEPADA, "Teskari Sanoq [ T ]" tugmasi uning PASTIDA ---
   ObjSetXY(prefix+"btn_retracement", px+10, py + 418); 
   ObjSetXY(prefix+"btn_cd_toggle", px+10, py + 450); 

   // --- Parallel koordinatalar hisobi ---
   int currentExtraOffset = 262; // Asosiy paneldan o'ngga surilish masofasi

   if(infoPanelOpen)
     {
      int ix = px + currentExtraOffset; 
      ObjSetXY(prefix+"info_bg", ix, py);
      ObjSetXY(prefix+"info_border", ix, py);
      ObjSetXY(prefix+"info_header", ix, py);
      ObjSetXY(prefix+"info_title", ix + 15, py + 14);

      int activeDisplayRows = g_orderCount;
      if (activeDisplayRows < 1)  activeDisplayRows = 1;
      if (activeDisplayRows > AZZO_MAX_GRID_ORDERS) activeDisplayRows = AZZO_MAX_GRID_ORDERS;

      // --- "Orderlar soni" maydoni ENDI YUQORIDA (Order qatorlaridan oldin) ---
      int ocY = py + 55;
      ObjSetXY(prefix+"info_oc_lbl", ix + 15, ocY);
      ObjSetXY(prefix+"info_ordercount", ix + 15, ocY + 18);

      // --- Order qatorlari (Order N : Lot, yoki guruhlangan N-M : Lot) ENDI
      // --- "Orderlar soni" blokidan PASTDA. Qatorlar soni/diapazoni BuildInfoRows()
      // --- orqali xuddi CreateDashboard() dagi bilan bir xil hisoblanadi.
      int lotsStartY = ocY + 58;
      {
       int rowStart[]; int rowEnd[];
       int rowCount = BuildInfoRows(activeDisplayRows, rowStart, rowEnd);

       for(int i = 0; i < rowCount; i++)
         {
          ObjSetXY(prefix+"info_lbl_"+(string)i, ix + 15, lotsStartY + (i * 22));
          ObjSetXY(prefix+"info_val_"+(string)i, ix + 95, lotsStartY - 2 + (i * 22));
          ObjSetXY(prefix+"info_unit_"+(string)i, ix + 155, lotsStartY + (i * 22));
         }
      }

      // --- YANGI: "ORDERLAR ORASI" (Grid Step) bloki - Lot qatorlaridan PASTDA,
      // --- lekin FIKS joyda (AZZO_MAX_INFO_ROWS=6 ta Lot qatoriga mo'ljallangan
      // --- joydan keyin) - shu bilan orderlar soni (demak Lot qatorlari soni)
      // --- qancha bo'lishidan qat'i nazar, bu blok hech qachon Lot qatorlari
      // --- bilan ustma-ust tushmaydi.
      int gridTitleY = lotsStartY + (AZZO_MAX_INFO_ROWS * 22) + 12;
      ObjSetXY(prefix+"info_grid_title", ix + 15, gridTitleY);

      int gridRowStart = gridTitleY + 20;
      int gridStep = 24;
      ObjSetXY(prefix+"info_grid_lbl_1", ix + 15, gridRowStart);
      ObjSetXY(prefix+"info_grid_val_1", ix + 130, gridRowStart - 2);
      ObjSetXY(prefix+"info_grid_unit_1", ix + 190, gridRowStart);

      ObjSetXY(prefix+"info_grid_lbl_2", ix + 15, gridRowStart + gridStep);
      ObjSetXY(prefix+"info_grid_val_2", ix + 130, gridRowStart + gridStep - 2);
      ObjSetXY(prefix+"info_grid_unit_2", ix + 190, gridRowStart + gridStep);

      ObjSetXY(prefix+"info_grid_lbl_3", ix + 15, gridRowStart + (gridStep*2));
      ObjSetXY(prefix+"info_grid_val_3", ix + 130, gridRowStart + (gridStep*2) - 2);
      ObjSetXY(prefix+"info_grid_unit_3", ix + 190, gridRowStart + (gridStep*2));

      // --- YANGI: "MASOFASI" va "SL" qatorlari - "2 GURUH ORASI" (index 2) dan
      // --- PASTDA, xuddi shu 24px panjara bo'yicha davom etadi ---
      ObjSetXY(prefix+"info_grid_lbl_4", ix + 15, gridRowStart + (gridStep*3));
      ObjSetXY(prefix+"info_grid_val_4", ix + 130, gridRowStart + (gridStep*3) - 2);
      ObjSetXY(prefix+"info_grid_unit_4", ix + 190, gridRowStart + (gridStep*3));

      ObjSetXY(prefix+"info_grid_lbl_5", ix + 15, gridRowStart + (gridStep*4));
      ObjSetXY(prefix+"info_grid_val_5", ix + 130, gridRowStart + (gridStep*4) - 2);
      ObjSetXY(prefix+"info_grid_unit_5", ix + 190, gridRowStart + (gridStep*4));

      // --- YANGI: TAYYOR SOZLAMALAR (PRESET) tugmalari - "SL" qatoridan PASTDA.
      // --- "NFP" va "CPI" bir qatorda yonma-yon (har biri 90px), "OTHER NEWS"
      // --- ularning tagida to'liq kenglikda (185px) - INFO panelning 215px
      // --- eniga (15px chap/o'ng margin bilan) aniq sig'adigan o'lchamlarda.
      int presetTitleY = gridRowStart + (gridStep*4) + 30;
      int presetRow1Y  = presetTitleY + 16;
      int presetRow2Y  = presetRow1Y + 24 + 8;

      ObjSetXY(prefix+"info_preset_title", ix + 15, presetTitleY);
      ObjSetXY(prefix+"info_btn_nfp",        ix + 15,  presetRow1Y);
      ObjSetXY(prefix+"info_btn_cpi",        ix + 110, presetRow1Y);
      ObjSetXY(prefix+"info_btn_othernews",  ix + 15,  presetRow2Y);

      currentExtraOffset += 217; // 182 -> 217: infoWidth 180 -> 215 ga kattalashtirilgani uchun keyingi panellar ham mos ravishda o'ngga suriladi
     }

   // --- RETRACEMENT (LIMIT ORDERS) paneli joylashuvi ---
   // MA'LUMOT (INFO) paneli bilan BIR XIL currentExtraOffset stack-ida, undan
   // KEYIN keladi (info ochiq bo'lsa - undan o'ngda, yopiq bo'lsa - asosiy
   // panelning o'zidan o'ngda). Shu bilan asosiy panel qayerga ko'chirilsa
   // (drag qilinsa yoki burchak o'zgartirilsa) ham, TO'LIQ SINXRON harakatlanadi.
   if(retracementPanelOpen)
     {
      int rx = px + currentExtraOffset;
      ObjSetXY(prefix+"retr_bg", rx, py);
      ObjSetXY(prefix+"retr_border", rx, py);
      ObjSetXY(prefix+"retr_header", rx, py);
      ObjSetXY(prefix+"retr_title", rx + 15, py + 14);

      // --- Qatorlar endi INFO paneldagi "ORDERLAR ORASI" qatorlari bilan AYNAN
      // --- BIR XIL joylashuvda: label va Edit maydon BIR QATORDA (label rx+15,
      // --- edit rx+130), tepadan-pastga 26px qadam bilan. "HOLAT :" qatori ENG
      // --- TEPADA (Trigger'dan OLDIN), qolgan 5 qator undan PASTDA ketma-ket.
      int rowStartY = py + 58;
      int rowStep    = 26;

      ObjSetXY(prefix+"retr_lbl_status", rx + 15, rowStartY);
      ObjSetXY(prefix+"retr_val_status", rx + 130, rowStartY - 2);

      for(int i = 0; i < 5; i++)
        {
         int ry = rowStartY + ((i+1) * rowStep);
         ObjSetXY(prefix+"retr_lbl_"+(string)i, rx + 15, ry);
         ObjSetXY(prefix+"retr_val_"+(string)i, rx + 130, ry - 2);
        }

      currentExtraOffset += 217; // 192 -> 217: retrWidth 180 -> 215 ga kattalashtirilgani (INFO bilan bir xil) uchun keyingi panellar ham mos ravishda o'ngga suriladi
     }

   // --- TESKARI SANOQ (COUNTDOWN) paneli joylashuvi ---
   // MUHIM: agar InpPanelCorner "Lower" (Left_Lower/Right_Lower) bo'lsa, asosiy
   // panel chartning ENG PASTIDA turadi va uning "pastida" joy chart chegarasidan
   // TASHQARIDA (ko'rinmaydigan joyda) qoladi. Shuning uchun:
   //   - "Lower" burchaklarda -> countdown paneli asosiy panelning USTIDA (yuqorisida)
   //   - "Upper" burchaklarda -> countdown paneli INFO'dan PASTIDA (avvalgidek)
   // Shunday qilib panel har doim chart ichida, ko'rinadigan joyda bo'ladi.
   if(countdownPanelOpen)
     {
      int cdWidth  = MAIN_PANEL_WIDTH;
      int cdHeight = 150; // YANA HAM KATTALASHTIRILDI (130 -> 150) - CreateDashboard bilan bir xil qiymat
      int cdY;

      bool isLowerCorner = (currentCorner == Corner_Left_Lower || currentCorner == Corner_Right_Lower);

      if(isLowerCorner)
        {
         cdY = py - cdHeight - 10; // asosiy panelning TEPASIGA chiqariladi
         if(cdY < 0) cdY = 0;      // chart chegarasidan chiqib ketmasligi uchun
        }
      else
        {
         int bottomY = py + MAIN_PANEL_HEIGHT;      // asosiy panel / INFO pastki cheti
         cdY = bottomY + 10;
        }

      ObjSetXY(prefix+"cd_bg", px, cdY);
      ObjSetXY(prefix+"cd_border", px, cdY);
      ObjSetXY(prefix+"cd_header", px, cdY);
      ObjSetXY(prefix+"cd_title", px + 15, cdY + 6);
      // ANCHOR_CENTER tufayli bu koordinata matnning ANIQ MARKAZI bo'ladi - shuning
      // uchun sarlavha (header, balandligi 26px) dan pastdagi qolgan maydonning
      // gorizontal VA vertikal markazi hisoblanadi.
      ObjSetXY(prefix+"cd_value", px + cdWidth/2, cdY + 26 + (cdHeight-26)/2);
     }

   // --- FOREX FACTORY (F.FACTORY) paneli joylashuvi ---
   // Asosiy panelning O'NG TOMONIGA (currentExtraOffset stack ichiga) qo'shiladi,
   // INFO paneli kabi. Har bir qatorda: chapda qizil "impact" chizig'i, undan keyin
   // valyuta "badge"i, sana/vaqt va yangilik nomi - saytdagi jadval tuzilishiga o'xshab.
   if(newsPanelOpen)
     {
      int newsRows = InpNewsCalendarCount;
      if(newsRows < 1) newsRows = 1;
      if(newsRows > NEWS_MAX_ROWS) newsRows = NEWS_MAX_ROWS;

      int newsWidth   = 320;
      int newsHeaderH = 30;
      int newsCdRowH  = 28;
      int newsRowH    = 40;
      int newsHeight  = newsHeaderH + newsCdRowH + (newsRows * newsRowH) + 10;

      int nx = px + currentExtraOffset;
      int newsY = py;

      // --- Chart o'ng chegarasidan chiqib ketmasligi uchun cheklov ---
      if(nx > chart_w - newsWidth) nx = chart_w - newsWidth;
      if(nx < 0) nx = 0;
      // --- Chart pastki chegarasidan chiqib ketmasligi uchun cheklov ---
      if(newsY > chart_h - newsHeight) newsY = chart_h - newsHeight;
      if(newsY < 0) newsY = 0;

      ObjSetXY(prefix+"news_bg", nx, newsY);
      ObjSetXY(prefix+"news_border", nx, newsY);
      ObjSetXY(prefix+"news_header", nx, newsY);
      ObjSetXY(prefix+"news_title", nx + 14, newsY + 8);

      int cdY = newsY + newsHeaderH;
      ObjSetXY(prefix+"news_cd_bg", nx, cdY);
      ObjSetXY(prefix+"news_cd_lbl", nx + 10, cdY + 7);
      ObjSetXY(prefix+"news_cd_val", nx + 140, cdY + 5);

      int rowStartY = newsY + newsHeaderH + newsCdRowH + 4;
      for(int i = 0; i < newsRows; i++)
        {
         int ry = rowStartY + (i * newsRowH);
         ObjSetXY(prefix+"news_stripe_"+(string)i, nx, ry);
         ObjSetXY(prefix+"news_impact_"+(string)i, nx, ry + 4);
         ObjSetXY(prefix+"news_curbg_"+(string)i, nx + 10, ry + 6);
         ObjSetXY(prefix+"news_cur_"+(string)i, nx + 15, ry + 9);
         ObjSetXY(prefix+"news_row1_"+(string)i, nx + 58, ry + 7);
         ObjSetXY(prefix+"news_row2_"+(string)i, nx + 10, ry + 21);
        }

      currentExtraOffset += newsWidth + 10; // agar F.FACTORY paneli ochiq bo'lsa, undan keyingi narsalar yana o'ngga suriladi
     }

  }

void ObjSetXY(string name, int x, int y)
  {
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
  }

void UpdateDashboard()
  {
   if(panelMinimized) return; 
   
   ObjectSetString(0, prefix+"val_sym", OBJPROP_TEXT, _Symbol);
   ObjectSetString(0, prefix+"val_bal", OBJPROP_TEXT, DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + " USD");
   ObjectSetString(0, prefix+"val_spd", OBJPROP_TEXT, (string)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) + " Pts");
   ObjectSetString(0, prefix+"val_lev", OBJPROP_TEXT, "1:" + (string)AccountInfoInteger(ACCOUNT_LEVERAGE));
   // MUHIM: val_trig endi OBJ_EDIT (tahrirlanadigan maydon). Uni bu yerda har
   // tikda qayta yozib qo'ysak, foydalanuvchi hali yozib turgan/saqlamagan
   // matnini ustidan bosib o'chirib yuboradi. Shu sabab matni faqat
   // CreateDashboard() da (ochilganda) va CHARTEVENT_OBJECT_ENDEDIT da (foydalanuvchi
   // Enter bosganda/joyni tashlab ketganda) yangilanadi - OnTick/OnTimer'da EMAS.
   ObjectSetString(0, prefix+"val_slp", OBJPROP_TEXT, (string)InpMaxSlippage + " Pts");
   
   ObjectSetString(0, prefix+"val_srv", OBJPROP_TEXT, TimeToString(TimeTradeServer(), TIME_SECONDS));
   ObjectSetString(0, prefix+"val_loc", OBJPROP_TEXT, TimeToString(TimeLocal(), TIME_SECONDS));
   
   int ping_ms = (int)(TerminalInfoInteger(TERMINAL_PING_LAST) / 1000);
   string ping_str = (ping_ms > 0) ? IntegerToString(ping_ms) + " ms" : "0 ms";
   ObjectSetString(0, prefix+"val_ping", OBJPROP_TEXT, ping_str);

   if(globalTradeNFP)
     {
      string activeStr = (robotLang == 3) ? "[АКТИВЕН]" : "[ACTIVE]";
      ObjectSetString(0, prefix+"status_txt", OBJPROP_TEXT, activeStr); 
      ObjectSetInteger(0, prefix+"status_txt", OBJPROP_COLOR, C'0,255,102'); 
     }
   else
     {
      string pausedStr = (robotLang == 1) ? "[BYPASS]" : ((robotLang == 3) ? "[BYPASS]" : "[BYPASS]");
      ObjectSetString(0, prefix+"status_txt", OBJPROP_TEXT, pausedStr); 
      ObjectSetInteger(0, prefix+"status_txt", OBJPROP_COLOR, C'255,60,60'); 
     }

   // --- AZZO ML: "YO'NALISH:" yozuvini yangilash (og'ir hisob-kitob 3 soniyada
   // --- bir marta AZZO_UpdateRegimeIfDue() ichida bo'ladi, bu yerda faqat
   // --- panel matni/rangi qo'yiladi - har tikda arzon operatsiya) ---
   AZZO_UpdateRegimeIfDue();
   ObjectSetString(0, prefix+"val_dir", OBJPROP_TEXT, g_dirText);
   ObjectSetInteger(0, prefix+"val_dir", OBJPROP_COLOR, g_dirColor);

   // --- YANGI: "MAKSIMAL LOT:" qatorini yangilash (arzon operatsiya - har tikda) ---
   double maxLot = AZZO_CalcMaxLot();
   int    maxLotDigits = (SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP) < 0.01) ? 3 : 2;
   ObjectSetString(0, prefix+"val_maxlot", OBJPROP_TEXT, DoubleToString(maxLot, maxLotDigits));
     
   ChartRedraw(0);
  }

void CloseAllPositionsAndOrders()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) == _Symbol) trade.PositionClose(PositionGetInteger(POSITION_TICKET));
     }
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      ulong oTicket = OrderGetTicket(i);
      if(oTicket > 0 && OrderGetString(ORDER_SYMBOL) == _Symbol) trade.OrderDelete(oTicket);
     }
   // --- RECOVERY GRID: hammasi qo'lda tozalanganda holatni ham reset qilamiz ---
   g_recoveryGridActive = false;
   g_recoveryDirection  = 0;
  }

void PlaceStraddleOrders()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   double baseBuyPrice = ask + (g_orderDistance * point);
   double baseSellPrice = bid - (g_orderDistance * point);

   double unifiedBuySL = (g_orderSL > 0) ? baseBuyPrice - (g_orderSL * point) : 0;
   double unifiedSellSL = (g_orderSL > 0) ? baseSellPrice + (g_orderSL * point) : 0;
   
   bool buySuccess = false; bool sellSuccess = false;
   int ordersToPlace = g_orderCount;
   
   for(int i = 0; i < ordersToPlace; i++)
     {
      double gridOffset = GetGridOffsetPoints(i);
      double buyPrice = baseBuyPrice + (gridOffset * point);
      double sellPrice = baseSellPrice - (gridOffset * point);
      
      double buyTP = (InpTakeProfit > 0) ? buyPrice + (InpTakeProfit * point) : 0;
      double sellTP = (InpTakeProfit > 0) ? sellPrice - (InpTakeProfit * point) : 0;
      
      ulong currentLevelMagic = MAGIC_NUMBER + (ulong)i;
      trade.SetExpertMagicNumber(currentLevelMagic);
      
      double stepLot = inputLots[i];
      
      if(trade.BuyStop(stepLot, NormalizeDouble(buyPrice, _Digits), _Symbol, NormalizeDouble(unifiedBuySL, _Digits), NormalizeDouble(buyTP, _Digits))) buySuccess = true;
      else PrintFormat("AZZO: BuyStop level %d rad etildi -> retcode=%d (%s)", i, trade.ResultRetcode(), trade.ResultRetcodeDescription());

      if(trade.SellStop(stepLot, NormalizeDouble(sellPrice, _Digits), _Symbol, NormalizeDouble(unifiedSellSL, _Digits), NormalizeDouble(sellTP, _Digits))) sellSuccess = true;
      else PrintFormat("AZZO: SellStop level %d rad etildi -> retcode=%d (%s)", i, trade.ResultRetcode(), trade.ResultRetcodeDescription());
     }
   if(buySuccess || sellSuccess)
     {
      ordersPlaced = true;
      if(InpTelegramEnabled && InpTelegramOnStraddle)
        {
         string lotsList = "";
         for(int i = 0; i < ordersToPlace; i++)
           {
            lotsList += DoubleToString(inputLots[i], 2);
            if(i < ordersToPlace - 1) lotsList += ", ";
           }

         string msg = StringFormat(
            "🎯 ORDER QO'YILDI!\n\n"
            "💱 PARA: %s\n"
            "🔢 SONI: %d ta order\n"
            "📊 LOT (har biri): %s\n"
            "💰 NARX: Ask %s / Bid %s\n"
            "⏰ VAQT: %s\n\n"
            "🚀 NFP JANGI BOSHLANDI, OMAD!",
            _Symbol, ordersToPlace, lotsList,
            DoubleToString(ask, _Digits), DoubleToString(bid, _Digits),
            FormatUzTime(TimeCurrent()));
         SendTelegramMessage(msg);
        }
     }
   trade.SetExpertMagicNumber(MAGIC_NUMBER);
  }

void PlaceManualOrders()
  {
   PlaceStraddleOrders();
  }

//+------------------------------------------------------------------+
//| PENDING ORDER ZONES - vizual chiziqlar                             |
//| ----------------------------------------------------------------- |
//| Buy Stop / Sell Stop qo'yiladigan narxlarni (InpDistance orqali,   |
//| masalan 750 pts = 7.5$; agar 800/900 qilsa - 8$/9$ masofa) hamda   |
//| shu orderlarning SL/TP darajalarini grafikda QALIN, QISQA (butun   |
//| grafikni egallamaydigan) trend-chiziqlar bilan ko'rsatadi:         |
//|   - Entry (Buy Stop / Sell Stop) narxi -> XIRALASHTIRILGAN SARIQ   |
//|     (InpZoneOpacity foizda, ko'zni olmasin deb to'qroq qilingan)   |
//|   - Stop Loss darajasi                 -> QIZIL                   |
//|   - Take Profit darajasi                -> YASHIL                 |
//| Har OnTick()'da (zonalar ko'rinib turgan bo'lsa) qayta chaqiriladi |
//| - shu orqali Bid/Ask o'zgarganda chiziqlar ham darhol siljiydi.    |
//+------------------------------------------------------------------+
color ColorBlend(color fg, color bg, double opacityPercent)
  {
   double a = opacityPercent / 100.0;
   if(a < 0.0) a = 0.0;
   if(a > 1.0) a = 1.0;

   // MQL5 'color' - COLORREF formatida (0x00BBGGRR): eng past bayt = R, keyingisi = G, so'ng B.
   int r1 = (int)(fg & 0xFF),        g1 = (int)((fg >> 8) & 0xFF),  b1 = (int)((fg >> 16) & 0xFF);
   int r2 = (int)(bg & 0xFF),        g2 = (int)((bg >> 8) & 0xFF),  b2 = (int)((bg >> 16) & 0xFF);

   int rr = (int)MathRound(r1 * a + r2 * (1.0 - a));
   int gg = (int)MathRound(g1 * a + g2 * (1.0 - a));
   int bb = (int)MathRound(b1 * a + b2 * (1.0 - a));

   rr = (int)MathMax(0, MathMin(255, rr));
   gg = (int)MathMax(0, MathMin(255, gg));
   bb = (int)MathMax(0, MathMin(255, bb));

   return (color)(rr + (gg << 8) + (bb << 16));
  }

// Bitta zona-chizig'ini yaratadi (birinchi marta) yoki mavjud bo'lsa faqat
// koordinatalarini/rangini yangilaydi (obyektni qayta o'chirib-yaratmaydi -
// shu bilan har tikda chaqirilsa ham chayqalish/miltillash bo'lmaydi).
// Chiziq HAR DOIM joriy (eng oxirgi) sham atrofida markazlashtiriladi:
// InpZoneCandlesSide ta sham CHAPGA va xuddi shuncha O'NGGA. Yangi sham
// ochilishi bilan barTime o'zi ilgarilaydi - shu bilan chiziq ham avtomatik
// "yana 12ta chapga, 12ta o'ngga" bo'lib siljib boradi.
void DrawZoneLine(string name, double price, color clr, int width, ENUM_LINE_STYLE style)
  {
   int periodSec = PeriodSeconds();
   if(periodSec <= 0) periodSec = 60;

   int side = InpZoneCandlesSide;
   if(side < 1) side = 1;

   datetime barTime = iTime(_Symbol, _Period, 0); // joriy (eng oxirgi) shamning ochilish vaqti
   if(barTime <= 0) barTime = TimeCurrent();

   datetime t1 = barTime - periodSec * side; // chapga (orqaga) N sham
   datetime t2 = barTime + periodSec * side; // o'ngga (oldinga) N sham

   if(ObjectFind(0, name) < 0)
     {
      ObjectCreate(0, name, OBJ_TREND, 0, t1, price, t2, price);
      ObjectSetInteger(0, name, OBJPROP_RAY_LEFT, false);
      ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_SELECTED, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, name, OBJPROP_BACK, false);
     }
   else
     {
      ObjectMove(0, name, 0, t1, price);
      ObjectMove(0, name, 1, t2, price);
     }

   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
  }

// Bid va Ask orasidagi (spread) bo'shliqni KULRANG, xiralashtirilgan (InpSpreadOpacity %%)
// to'ldirilgan to'rtburchak bilan bo'yaydi. Chap tomoni CHEKSIZ (juda uzoq o'tmishga
// qadar cho'ziladi - MT5'da OBJ_RECTANGLE RAY_LEFT'ni QOLLAMAYDI, shuning uchun vaqt
// ankorini qasddan juda uzoq o'tmishga qo'yamiz), o'ng tomoni esa hozirgi (eng oxirgi)
// shamda to'xtaydi - Buy/Sell Stop/SL/TP zonalari bilan HECH QACHON aralashmaydi.
// MUHIM: bu funksiya "P" (Pending Order Zones) tugmasidan MUSTAQIL - robot chart'ga
// ulangan zahoti avtomatik chiziladi va "P" bosilsa ham O'CHMAYDI/O'CHIRIB BO'LMAYDI.
void DrawSpreadZone()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   string name = prefix + "spreadzone"; // "zone_" bilan BOSHLANMAYDI - shu bilan DeleteAllZoneLines() (P) bunga tegmaydi

   if(ask <= 0 || bid <= 0 || ask <= bid)
     {
      ObjectDelete(0, name);
      return;
     }

   int periodSec = PeriodSeconds();
   if(periodSec <= 0) periodSec = 60;

   datetime barTime = iTime(_Symbol, _Period, 0);
   if(barTime <= 0) barTime = TimeCurrent();

   datetime t1 = D'2000.01.01 00:00:00'; // "cheksiz chap" effekti - istalgan real grafikdan ancha uzoqdagi o'tmish
   datetime t2 = barTime + periodSec;    // joriy (eng oxirgi) shamning o'ng cheti - bundan o'ngga ASLO o'tmaydi

   color spreadColor = ColorBlend(clrSilver, clrBlack, InpSpreadOpacity);

   if(ObjectFind(0, name) < 0)
     {
      ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, ask, t2, bid);
      ObjectSetInteger(0, name, OBJPROP_FILL, true);
      ObjectSetInteger(0, name, OBJPROP_BACK, true);       // shamlar/svechalar ustidan bosib qolmasin
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_SELECTED, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
     }
   else
     {
      ObjectMove(0, name, 0, t1, ask);
      ObjectMove(0, name, 1, t2, bid);
     }

   ObjectSetInteger(0, name, OBJPROP_COLOR, spreadColor);
  }

// Joriy Ask/Bid, InpDistance, InpGridStepGroup1/InpGridStepGroup2/
// InpGridGapBetweenGroups, InpStopLoss, InpTakeProfit va g_orderCount asosida
// BARCHA daraja (level) uchun Entry/SL/TP zonalarini chizadi yoki (agar
// allaqachon chizilgan bo'lsa) yangi narxga siljitadi.
void DrawPendingZones()
  {
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0) return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double baseBuyPrice  = ask + (g_orderDistance * point);
   double baseSellPrice = bid - (g_orderDistance * point);

   double unifiedBuySL  = (g_orderSL > 0) ? baseBuyPrice  - (g_orderSL * point) : 0;
   double unifiedSellSL = (g_orderSL > 0) ? baseSellPrice + (g_orderSL * point) : 0;

   color zoneEntryColor = ColorBlend(clrGold, clrBlack, InpZoneOpacity); // sariq, ko'zni olmaydigan to'q ton
   color zoneSLColor    = clrRed;
   color zoneTPColor    = clrLime;

   int levels = g_orderCount;
   if(levels < 1)  levels = 1;
   if(levels > AZZO_MAX_GRID_ORDERS) levels = AZZO_MAX_GRID_ORDERS;

   for(int i = 0; i < levels; i++)
     {
      double gridOffset = GetGridOffsetPoints(i);
      double buyPrice  = baseBuyPrice  + (gridOffset * point);
      double sellPrice = baseSellPrice - (gridOffset * point);

      double buyTP  = (InpTakeProfit > 0) ? buyPrice  + (InpTakeProfit * point) : 0;
      double sellTP = (InpTakeProfit > 0) ? sellPrice - (InpTakeProfit * point) : 0;

      DrawZoneLine(prefix + "zone_buyE_"  + (string)i, buyPrice,      zoneEntryColor, InpZoneWidth, STYLE_SOLID);
      DrawZoneLine(prefix + "zone_sellE_" + (string)i, sellPrice,     zoneEntryColor, InpZoneWidth, STYLE_SOLID);

      if(unifiedBuySL > 0)  DrawZoneLine(prefix + "zone_buySL_"  + (string)i, unifiedBuySL,  zoneSLColor, InpZoneWidth, STYLE_SOLID);
      else                  ObjectDelete(0, prefix + "zone_buySL_"  + (string)i);

      if(unifiedSellSL > 0) DrawZoneLine(prefix + "zone_sellSL_" + (string)i, unifiedSellSL, zoneSLColor, InpZoneWidth, STYLE_SOLID);
      else                  ObjectDelete(0, prefix + "zone_sellSL_" + (string)i);

      if(buyTP > 0)  DrawZoneLine(prefix + "zone_buyTP_"  + (string)i, buyTP,  zoneTPColor, InpZoneWidth, STYLE_SOLID);
      else           ObjectDelete(0, prefix + "zone_buyTP_"  + (string)i);

      if(sellTP > 0) DrawZoneLine(prefix + "zone_sellTP_" + (string)i, sellTP, zoneTPColor, InpZoneWidth, STYLE_SOLID);
      else           ObjectDelete(0, prefix + "zone_sellTP_" + (string)i);
     }

   // Agar oldin orderlar soni kattaroq bo'lib, keyin kamaytirilgan bo'lsa -
   // ortiqcha (endi ishlatilmayotgan) yuqori darajalar chizig'ini tozalaymiz.
   for(int i = levels; i < AZZO_MAX_GRID_ORDERS; i++)
     {
      ObjectDelete(0, prefix + "zone_buyE_"   + (string)i);
      ObjectDelete(0, prefix + "zone_sellE_"  + (string)i);
      ObjectDelete(0, prefix + "zone_buySL_"  + (string)i);
      ObjectDelete(0, prefix + "zone_sellSL_" + (string)i);
      ObjectDelete(0, prefix + "zone_buyTP_"  + (string)i);
      ObjectDelete(0, prefix + "zone_sellTP_" + (string)i);
     }

   // Bid/Ask oralig'idagi kulrang spread-zonasi endi MUSTAQIL (P ga bog'liq emas) -
   // DrawSpreadZone() OnInit()/OnTick() ichida alohida chaqiriladi, shu yerda emas.

   ChartRedraw(0);
  }

// "P" bilan o'chirilganda (yoki EA to'xtaganda) barcha zona-chiziqlarini tozalaydi.
void DeleteAllZoneLines()
  {
   ObjectsDeleteAll(0, prefix + "zone_");
   ChartRedraw(0);
  }

void CheckSmartOCO()
  {
   if(!InpUseOCO) return;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) == _Symbol)
        {
         ulong posMagic = PositionGetInteger(POSITION_MAGIC);
         if(posMagic >= MAGIC_NUMBER && posMagic <= MAGIC_NUMBER + (ulong)g_orderCount)
           {
            for(int j = OrdersTotal() - 1; j >= 0; j--)
              {
               ulong orderTicket = OrderGetTicket(j);
               if(orderTicket > 0 && OrderGetString(ORDER_SYMBOL) == _Symbol)
                 {
                  if(OrderGetInteger(ORDER_MAGIC) == posMagic) trade.OrderDelete(orderTicket);
                 }
              }
           }
        }
     }
  }

void ApplyTrailingStop()
  {
   if(!InpUseTrailing) return;
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) == _Symbol)
        {
         ulong posMagic = PositionGetInteger(POSITION_MAGIC);
         if(posMagic >= MAGIC_NUMBER && posMagic <= MAGIC_NUMBER + (ulong)g_orderCount)
           {
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentSL = PositionGetDouble(POSITION_SL);
            
            if(type == POSITION_TYPE_BUY)
              {
               double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
               if(bid - openPrice > InpTrailingStart * point)
                 {
                  double newSL = bid - InpTrailingStep * point;
                  if(currentSL == 0 || newSL > currentSL + point) trade.PositionModify(ticket, NormalizeDouble(newSL, _Digits), PositionGetDouble(POSITION_TP));
                 }
              }
            else if(type == POSITION_TYPE_SELL)
              {
               double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               if(openPrice - ask > InpTrailingStart * point)
                 {
                  double newSL = ask + InpTrailingStep * point;
                  if(currentSL == 0 || newSL < currentSL - point) trade.PositionModify(ticket, NormalizeDouble(newSL, _Digits), PositionGetDouble(POSITION_TP));
                 }
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| RECOVERY / AVERAGING GRID  --  SODDA TUSHUNTIRISH                  |
//| -------------------------------------------------------------------|
//| 3 TA BOSQICH, MISOL BILAN (narx = 4000$):                          |
//|                                                                     |
//|  1) ASOSIY GRID MASOFASI (InpDistance, masalan 750 punkt = 7.5$)   |
//|     -> Origin (NFP narxi) dan 7.5$ narida entry ochilgan:          |
//|        4000$ + 7.5$ = 4007.5$ (bu ENTRY narxi)                    |
//|                                                                     |
//|  2) TRIGGER (InpRecoveryTriggerPoints, masalan 200 punkt = 2$)     |
//|     -> ENTRY narxidan yana 2$ FOYDA tomonga siljisa, Recovery      |
//|        ISHGA TUSHADI (faollashadi):                                |
//|        4007.5$ + 2$ = 4009.5$ da faollashish nuqtasi               |
//|        (Origindan hisoblasak: 7.5$ + 2$ = 9.5$ siljigan bo'ladi)   |
//|                                                                     |
//|  3) BIRINCHI DARAJA MASOFASI (InpRecoveryOffsetPoints, masalan     |
//|     100 punkt = 1$) -> Recovery ishga tushgan ONDA, JORIY narxdan  |
//|     yana 1$ narida BIRINCHI pending order qo'yiladi:                |
//|        4009.5$ + 1$ = 4010.5$ da BIRINCHI Buy Stop turadi          |
//|                                                                     |
//|  Shundan keyin har bir keyingi order QADAM (InpRecoveryStepPoints, |
//|  masalan 70 punkt = 0.70$) bilan navbatma-navbat uzoqlashadi:      |
//|     2-daraja: 4010.5$ + 0.70$ = 4011.20$                           |
//|     3-daraja: 4011.20$ + 0.70$ = 4011.90$   ... va hokazo           |
//|     (InpRecoveryLevels dona, HAR TOMONGA - yuqoriga ham, pastga ham)|
//|                                                                     |
//|  ESLATMA: 100 punkt = 1$ (XAUUSD uchun standart nisbat)            |
//|                                                                     |
//| YO'NALISH MANTIG'I (MT5 pending order qoidalariga mos):            |
//|   SELL grid: PASTDA -> Sell Stop (harakatni davom ettiradi),       |
//|              TEPADA -> Sell Limit (orqaga qaytishni "poylaydi")    |
//|   BUY  grid: TEPADA -> Buy Stop  (harakatni davom ettiradi),       |
//|              PASTDA -> Buy Limit (orqaga qaytishni "poylaydi")     |
//|                                                                     |
//| MUHIM: bu orderlar SL/TP'siz qo'yiladi (sof "averaging" maqsadida) |
//| - avval albatta DEMO hisobda sinab ko'ring.                        |
//+------------------------------------------------------------------+
void ManageRecoveryGrid()
  {
   if(!InpRecoveryEnable) return;

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0) return;

   int    dir            = 0;     // 1 = BUY, 2 = SELL
   int    totalPositions = 0;
   double refOpenBuy     = 0.0;   // BUY tomon uchun eng dastlabki (eng past) entry narxi
   double refOpenSell    = 0.0;   // SELL tomon uchun eng dastlabki (eng yuqori) entry narxi

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) != _Symbol) continue;
      ulong posMagic = PositionGetInteger(POSITION_MAGIC);
      // --- Faqat shu EA'ga tegishli orderlar (asosiy grid + recovery grid diapazoni) ---
      if(posMagic < MAGIC_NUMBER || posMagic > MAGIC_NUMBER + AZZO_RECOVERY_MAGIC_OFFSET + (ulong)(g_recLevels * 2 + 5)) continue;

      totalPositions++;
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);

      if(type == POSITION_TYPE_BUY)
        {
         dir = 1;
         if(refOpenBuy == 0.0 || openPrice < refOpenBuy) refOpenBuy = openPrice; // eng birinchi/eng qulay entry
        }
      else if(type == POSITION_TYPE_SELL)
        {
         dir = 2;
         if(refOpenSell == 0.0 || openPrice > refOpenSell) refOpenSell = openPrice; // eng birinchi/eng qulay entry
        }
     }

   // --- Ochiq pozitsiya umuman yo'q -> recovery holatini butunlay tozalaymiz ---
   if(totalPositions == 0)
     {
      if(g_recoveryGridActive) DeleteRecoveryPendingOrders();
      g_recoveryGridActive = false;
      g_recoveryDirection  = 0;
      return;
     }

   // --- Yo'nalish avvalgisidan farq qilsa (masalan grid tozalanib, qayta yangi tomonga ochilgan) ---
   if(g_recoveryDirection != 0 && g_recoveryDirection != dir)
     {
      DeleteRecoveryPendingOrders();
      g_recoveryGridActive = false;
     }
   g_recoveryDirection = dir;

   if(g_recoveryGridActive) return; // shu tsikl uchun allaqachon qo'yilgan

   double profitPoints = 0.0;
   if(dir == 1) // BUY: bid - openPrice
     {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      profitPoints = (bid - refOpenBuy) / point;
     }
   else if(dir == 2) // SELL: openPrice - ask
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      profitPoints = (refOpenSell - ask) / point;
     }
   else return;

   // --- SODDA: profitPoints = ENTRY narxidan hozirgi FOYDA (punktda).
   // --- Agar bu InpRecoveryTriggerPoints (masalan 200 = $2) dan kichik bo'lsa,
   // --- hali vaqti kelmagan, kutamiz.
   if(profitPoints < g_recTrigger) return; // hali trigger yetmagan

   PlaceRecoveryGrid(dir, profitPoints);
  }

void PlaceRecoveryGrid(int dir, double currentProfitPoints)
  {
   double ask   = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid   = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int    levels = g_recLevels;
   if(levels < 1) levels = 1;

   // --- 2-BOSQICH / HOLAT tugmasi: g_recBlockStopOrders = TRUE bo'lsa -> Recovery Grid
   // --- ISHLAYDI (Stop VA Limit orderlar qo'yiladi). FALSE bo'lsa -> HECH QANDAY
   // --- recovery order qo'yilmaydi (to'liq o'chirilgan holat).
   bool allowStop  = g_recBlockStopOrders;
   bool allowLimit = g_recBlockStopOrders;
   if(!allowStop && !allowLimit) return; // HOLAT = FALSE -> hech qanday recovery order qo'yilmaydi

   trade.SetExpertMagicNumber(MAGIC_NUMBER + AZZO_RECOVERY_MAGIC_OFFSET);

   int placedCount = 0;
   for(int i = 0; i < levels; i++)
     {
      // --- SODDA: i=0 (BIRINCHI daraja) -> masofa = faqat InpRecoveryOffsetPoints
      // --- (masalan 100 = $1). i=1,2,3... uchun har safar +InpRecoveryStepPoints
      // --- (masalan 70 = $0.70) qo'shilib boradi. Bu masofa JORIY narxdan
      // --- (ask/bid) hisoblanadi, entry yoki origin narxidan EMAS.
      double dist = (g_recOffset + (i * g_recStep)) * point;

      if(dir == 2) // --- SELL GRID ---
        {
         if(allowStop)
           {
            double stopPrice = NormalizeDouble(bid - dist, _Digits); // pastda -> Sell Stop
            if(trade.SellStop(g_recLot, stopPrice, _Symbol, 0, 0, ORDER_TIME_GTC, 0, "AzzoRecoverySell")) placedCount++;
           }
         if(allowLimit)
           {
            double limitPrice = NormalizeDouble(ask + dist, _Digits); // tepada -> Sell Limit
            if(trade.SellLimit(g_recLot, limitPrice, _Symbol, 0, 0, ORDER_TIME_GTC, 0, "AzzoRecoverySell")) placedCount++;
           }
        }
      else if(dir == 1) // --- BUY GRID ---
        {
         if(allowStop)
           {
            double stopPrice = NormalizeDouble(ask + dist, _Digits); // tepada -> Buy Stop
            if(trade.BuyStop(g_recLot, stopPrice, _Symbol, 0, 0, ORDER_TIME_GTC, 0, "AzzoRecoveryBuy")) placedCount++;
           }
         if(allowLimit)
           {
            double limitPrice = NormalizeDouble(bid - dist, _Digits); // pastda -> Buy Limit
            if(trade.BuyLimit(g_recLot, limitPrice, _Symbol, 0, 0, ORDER_TIME_GTC, 0, "AzzoRecoveryBuy")) placedCount++;
           }
        }
     }

   trade.SetExpertMagicNumber(MAGIC_NUMBER);

   if(placedCount > 0)
     {
      g_recoveryGridActive = true;

      if(InpTelegramEnabled)
        {
         string dirText = (dir == 1) ? "BUY" : "SELL";
         string modeText = g_recBlockStopOrders ? "Stop + Limit orderlar (faol)" : "BLOKLANGAN (hech qanday order qo'yilmaydi)";
         string msg = StringFormat(
            "♻️ RECOVERY GRID FAOLLASHDI!\n\n"
            "💱 PARA: %s\n"
            "📍 YO'NALISH: %s\n"
            "🔒 REJIM: %s\n"
            "💰 ENTRY'DAN SILJISH: %d punkt (trigger: %d punkt)\n"
            "🎯 DARAJALAR: %d ta (har tomonda)\n"
            "📏 QADAM: %d punkt | BOSHLANG'ICH MASOFA: %d punkt\n"
            "📦 LOT (har biri): %s\n"
            "⏰ VAQT: %s",
            _Symbol, dirText, modeText,
            (int)MathRound(currentProfitPoints), g_recTrigger,
            levels,
            g_recStep, g_recOffset,
            DoubleToString(g_recLot, 2),
            FormatUzTime(TimeCurrent()));
         SendTelegramMessage(msg);
        }
     }
  }

void DeleteRecoveryPendingOrders()
  {
   for(int j = OrdersTotal() - 1; j >= 0; j--)
     {
      ulong oTicket = OrderGetTicket(j);
      if(oTicket > 0 && OrderGetString(ORDER_SYMBOL) == _Symbol)
        {
         ulong oMagic = OrderGetInteger(ORDER_MAGIC);
         if(oMagic == MAGIC_NUMBER + AZZO_RECOVERY_MAGIC_OFFSET) trade.OrderDelete(oTicket);
        }
     }
  }

//+------------------------------------------------------------------+
//| UpdateCountdownPanel()                                             |
//| "TESKARI SANOQ" panelidagi qiymatni va rangini yangilaydi.        |
//| Joriy NFP maqsad vaqti (g_nfpTime) bilan broker vaqti orasidagi   |
//| qolgan vaqtni hisoblaydi:                                          |
//|   - qolgan vaqt > 60 sek  -> odatiy neon rang                     |
//|   - qolgan vaqt <= 60 sek -> SARIQ (yellow)                       |
//|   - qolgan vaqt <= 10 sek -> ORANGE                                |
//|   - qolgan vaqt <= 5  sek -> QIZIL (red)                          |
//|   - NFP vaqti keldi (trigger oynasi ichida) -> "BOOOM!" (YASHIL)  |
//| Trigger oynasi (InpTimeWindowSeconds) tugagach, avtomatik ravishda |
//| ERTANGI kunning xuddi shu vaqtiga qarab qayta sanoq boshlanadi.    |
//|                                                                     |
//| MATN FORMATI ("aqlli", ortiqcha nollarsiz):                       |
//|   - soat > 0   ->  "H:MM:SS"   (masalan "1:00:10")                 |
//|   - soat == 0, daqiqa > 0 -> "M:SS"      (masalan "59:59", "10:10")|
//|   - soat == 0, daqiqa == 0 -> "S"        (masalan "59", "5")       |
//+------------------------------------------------------------------+
void UpdateCountdownPanel()
  {
   if(!countdownPanelOpen) return;
   if(panelMinimized) return;
   if(ObjectFind(0, prefix+"cd_value") < 0) return; // panel hali chizilmagan bo'lsa, chiqib ketamiz

   datetime brokerTime = TimeTradeServer();
   datetime curr       = TimeCurrent();
   string   date_str   = TimeToString(curr, TIME_DATE);
   datetime targetTime = StringToTime(date_str + " " + g_nfpTime);

   // --- NFP trigger oynasi ichidamizmi? (xuddi OnTimer()'dagi PlaceStraddleOrders shartiga o'xshash) ---
   bool boomActive = (brokerTime >= targetTime && brokerTime <= targetTime + InpTimeWindowSeconds);

   if(boomActive)
     {
      ObjectSetString(0, prefix+"cd_value", OBJPROP_TEXT, "BOOOM!");
      ObjectSetInteger(0, prefix+"cd_value", OBJPROP_COLOR, C'0,255,102'); // YASHIL
      ObjectSetInteger(0, prefix+"cd_value", OBJPROP_FONTSIZE, 40);
      return;
     }

   long remaining = (long)(targetTime - brokerTime);

   // --- Bugungi trigger oynasi allaqachon o'tib ketgan bo'lsa -> ERTANGI kunning
   // --- xuddi shu vaqtiga qarab sanoq boshlanadi.
   if(remaining < 0)
     {
      targetTime += 86400; // +1 kun (soniyada)
      remaining = (long)(targetTime - brokerTime);
     }
   if(remaining < 0) remaining = 0; // xavfsizlik uchun chegara

   int hh = (int)(remaining / 3600);
   int mm = (int)((remaining % 3600) / 60);
   int ss = (int)(remaining % 60);

   // --- "Aqlli" format: yetakchi (bo'sh/nol) birliklar butunlay olib tashlanadi ---
   string cdText;
   if(hh > 0)
      cdText = StringFormat("%d:%02d:%02d", hh, mm, ss);   // masalan "1:00:10"
   else if(mm > 0)
      cdText = StringFormat("%d:%02d", mm, ss);             // masalan "59:59", "10:10"
   else
      cdText = StringFormat("%d", ss);                      // masalan "59", "5"

   color cdColor;
   if(remaining <= 5)       cdColor = clrRed;        // 5 soniya va undan kam
   else if(remaining <= 10) cdColor = clrOrange;     // 10 soniya va undan kam
   else if(remaining <= 60) cdColor = clrYellow;     // 1 daqiqa (60 sek) va undan kam
   else                     cdColor = ColorMainNeon; // odatiy holat

   ObjectSetString(0, prefix+"cd_value", OBJPROP_TEXT, cdText);
   ObjectSetInteger(0, prefix+"cd_value", OBJPROP_COLOR, cdColor);
   ObjectSetInteger(0, prefix+"cd_value", OBJPROP_FONTSIZE, 36);
  }

//+------------------------------------------------------------------+
//| UpdateMinimizedCountdown()                                        |
//| Panel "-" tugmasi (yoki "S" tez tugmasi) bilan kichraytirilganda   |
//| (panelMinimized == true) ko'rinadigan kichik satrdagi TESKARI      |
//| SANOQ qiymatini yangilaydi. Mantiq UpdateCountdownPanel() bilan    |
//| AYNAN BIR XIL (target NFP vaqti, "BOOOM!", rang bosqichlari),      |
//| shu sabab countdownPanelOpen (katta countdown paneli) ochiq yoki   |
//| yopiqligidan QAT'I NAZAR, panel minimallashtirilgan bo'lsa doim    |
//| to'g'ri qiymat ko'rsatiladi.                                       |
//+------------------------------------------------------------------+
void UpdateMinimizedCountdown()
  {
   if(!panelMinimized) return;
   if(ObjectFind(0, prefix+"min_cd_txt") < 0) return; // panel hali chizilmagan bo'lsa, chiqib ketamiz

   datetime brokerTime = TimeTradeServer();
   datetime curr       = TimeCurrent();
   string   date_str   = TimeToString(curr, TIME_DATE);
   datetime targetTime = StringToTime(date_str + " " + g_nfpTime);

   bool boomActive = (brokerTime >= targetTime && brokerTime <= targetTime + InpTimeWindowSeconds);

   if(boomActive)
     {
      ObjectSetString(0, prefix+"min_cd_txt", OBJPROP_TEXT, "BOOOM!");
      ObjectSetInteger(0, prefix+"min_cd_txt", OBJPROP_COLOR, C'0,255,102'); // YASHIL
      return;
     }

   long remaining = (long)(targetTime - brokerTime);
   if(remaining < 0)
     {
      targetTime += 86400; // +1 kun
      remaining = (long)(targetTime - brokerTime);
     }
   if(remaining < 0) remaining = 0;

   int hh = (int)(remaining / 3600);
   int mm = (int)((remaining % 3600) / 60);
   int ss = (int)(remaining % 60);

   string cdText;
   if(hh > 0)
      cdText = StringFormat("%d:%02d:%02d", hh, mm, ss);
   else if(mm > 0)
      cdText = StringFormat("%d:%02d", mm, ss);
   else
      cdText = StringFormat("%d", ss);

   color cdColor;
   if(remaining <= 5)       cdColor = clrRed;
   else if(remaining <= 10) cdColor = clrOrange;
   else if(remaining <= 60) cdColor = clrYellow;
   else                     cdColor = ColorMainNeon;

   ObjectSetString(0, prefix+"min_cd_txt", OBJPROP_TEXT, cdText);
   ObjectSetInteger(0, prefix+"min_cd_txt", OBJPROP_COLOR, cdColor);
  }

//+------------------------------------------------------------------+
//| CheckNfpPreAlert()                                                 |
//| NFP maqsad vaqtiga (g_nfpTime) InpPreAlertMinutes daqiqa qolganda   |
//| BIR MARTA (har bir NFP kuni uchun faqat bitta xabar):                |
//|   1) MT5 ICHIDA - Alert() orqali popup oyna + tovushli signal        |
//|      (Terminal ochiq bo'lsa darhol ko'rinadi, Journal/Experts        |
//|      loglariga ham yoziladi).                                        |
//|   2) TELEGRAM BOTGA - mavjud SendTelegramMessage() funksiyasi        |
//|      orqali (xuddi order ochilish/yopilish xabarlari kabi).          |
//| Xabar matni to'liq InpPreAlertMessage inputidan olinadi - istalgan   |
//| matnni yozishingiz mumkin, kod ichida HECH NARSA qattiq yozilmagan.  |
//| g_lastPreAlertTargetTime orqali bitta NFP maqsad vaqti uchun FAQAT   |
//| BIR MARTA ishga tushishi kafolatlanadi (har OnTimer tikida qayta-    |
//| qayta yubormaslik uchun).                                            |
//+------------------------------------------------------------------+
void CheckNfpPreAlert()
  {
   if(!InpPreAlertEnabled) return;

   datetime brokerTime = TimeTradeServer();
   datetime curr       = TimeCurrent();
   string   date_str   = TimeToString(curr, TIME_DATE);
   datetime targetTime = StringToTime(date_str + " " + g_nfpTime);

   long remaining = (long)(targetTime - brokerTime);
   if(remaining < 0)
     {
      targetTime += 86400; // bugungi trigger o'tib ketgan - ertangi kunning xuddi shu vaqtiga qarab hisoblanadi
      remaining = (long)(targetTime - brokerTime);
     }

   long preAlertSeconds = (long)InpPreAlertMinutes * 60;

   // --- Faqat "InpPreAlertMinutes" oynasi ICHIGA kirganda (masalan 5:00 va 0:00 orasida)
   // --- VA aynan shu targetTime uchun hali ogohlantirilmagan bo'lsa ishga tushadi.
   if(remaining <= preAlertSeconds && remaining > 0 && g_lastPreAlertTargetTime != targetTime)
     {
      g_lastPreAlertTargetTime = targetTime;

      Alert(InpPreAlertMessage);          // MT5 terminalida popup + tovushli ogohlantirish
      SendTelegramMessage(InpPreAlertMessage); // Telegram botga (agar InpTelegramEnabled=true bo'lsa)
     }
  }

//+------------------------------------------------------------------+
//| IsNewsCurrencyWanted()                                             |
//| InpNewsCurrencies inputida (vergul bilan ajratilgan, masalan       |
//| "NZD,USD") ko'rsatilgan valyutalar ro'yxatidan curCode borligini   |
//| tekshiradi. Katta-kichik harf va bo'shliqlarga sezgir emas.        |
//| InpNewsCurrencies BO'SH qoldirilsa - HAMMASI ruxsat (filtr yo'q).  |
//+------------------------------------------------------------------+
bool IsNewsCurrencyWanted(string curCode)
  {
   string filterRaw = InpNewsCurrencies;
   StringTrimLeft(filterRaw);
   StringTrimRight(filterRaw);
   if(filterRaw == "") return true; // filtr bo'sh - hamma valyuta ko'rsatiladi

   string wantedList[];
   int wantedCount = StringSplit(filterRaw, ',', wantedList);

   string curUpper = curCode;
   StringTrimLeft(curUpper);
   StringTrimRight(curUpper);
   StringToUpper(curUpper);

   for(int i = 0; i < wantedCount; i++)
     {
      string tok = wantedList[i];
      StringTrimLeft(tok);
      StringTrimRight(tok);
      StringToUpper(tok);
      if(tok == "") continue;
      if(tok == curUpper) return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| DayNameFF()                                                        |
//| ForexFactory saytidagi HAFTA KUNI formatiga mos (Mon, Tue, Wed,    |
//| Thu, Fri, Sat, Sun) qisqa INGLIZCHA nom qaytaradi. Til (robotLang) |
//| qanday bo'lishidan qat'i nazar F-panelidagi kun ForexFactory       |
//| saytidagi bilan bir xil ko'rinishda chiqishi uchun til ATAYLAB     |
//| e'tiborga olinmaydi.                                                |
//+------------------------------------------------------------------+
string DayNameFF(datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   string names[7] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
   int idx = dt.day_of_week;
   if(idx < 0 || idx > 6) return "";
   return names[idx];
  }

//+------------------------------------------------------------------+
//| RefreshNewsCalendar()                                              |
//| MT5 terminalining ICHKI Economic Calendar bazasidan (Calendar*     |
//| funksiyalari orqali) TOP ("High"/muhim) yangiliklarni o'qiydi.     |
//| Bu 100% MAHALLIY chaqiruv - hech qanday WebRequest yoki tarmoq      |
//| so'rovi YO'Q, shuning uchun URL ruxsat ro'yxatiga qo'shish shart    |
//| emas va bu funksiya HECH QACHON bloklanib/qotib qolmaydi. Terminal  |
//| kalendar bazasini fon rejimida o'zi avtomatik sinxronlab turadi.    |
//| Vaqt bo'yicha o'sish tartibida saralaydi va birinchi                |
//| InpNewsCalendarCount tasini g_news* massivlariga yozadi. Faqat      |
//| HALI BO'LMAGAN (kelajakdagi) yangiliklar olinadi, shuning uchun     |
//| ro'yxatning [0] elementi doim ENG YAQIN yangilikdir.                 |
//+------------------------------------------------------------------+
void RefreshNewsCalendar()
  {
   g_newsCount = 0;

   datetime fromTime = TimeTradeServer();
   int lookAheadDays = InpNewsLookAheadDays;
   if(lookAheadDays < 1) lookAheadDays = 1;
   datetime toTime = fromTime + (datetime)lookAheadDays * 86400;

   MqlCalendarValue values[];
   ResetLastError();
   int total = CalendarValueHistory(values, fromTime, toTime, NULL, NULL);
   if(total < 0)
     {
      int err = GetLastError();
      Print("AzzoTrade Calendar: MT5 ichki kalendaridan o'qishda xato (", err, "). ",
            "Terminal hali kalendar bazasini fon rejimida sinxronlab bo'lmagan bo'lishi mumkin - keyingi tikda qayta urinib ko'riladi.");
      g_newsDataReady = false;
      return;
     }

   datetime tTimes[]; string tNames[]; string tCur[];
   ArrayResize(tTimes, total);
   ArrayResize(tNames, total);
   ArrayResize(tCur,   total);
   int cnt = 0;

   for(int i = 0; i < total; i++)
     {
      MqlCalendarEvent evt;
      if(!CalendarEventById(values[i].event_id, evt)) continue;
      if(evt.importance != CALENDAR_IMPORTANCE_HIGH) continue; // faqat TOP/qizil (High) yangiliklar

      datetime evTime = values[i].time;
      if(evTime < fromTime) continue; // faqat kelajakdagi yangiliklar

      MqlCalendarCountry country;
      string curCode = "";
      if(CalendarCountryById(evt.country_id, country)) curCode = country.currency;

      if(!IsNewsCurrencyWanted(curCode)) continue; // InpNewsCurrencies ro'yxatida yo'q - o'tkazib yuborish

      tTimes[cnt] = evTime;
      tNames[cnt] = evt.name;
      tCur[cnt]   = curCode;
      cnt++;
     }

   // --- Vaqt bo'yicha o'sish tartibida saralash (insertion sort - ro'yxat odatda kichik) ---
   for(int i = 1; i < cnt; i++)
     {
      datetime keyT = tTimes[i]; string keyN = tNames[i]; string keyC = tCur[i];
      int j = i - 1;
      while(j >= 0 && tTimes[j] > keyT)
        {
         tTimes[j+1] = tTimes[j]; tNames[j+1] = tNames[j]; tCur[j+1] = tCur[j];
         j--;
        }
      tTimes[j+1] = keyT; tNames[j+1] = keyN; tCur[j+1] = keyC;
     }

   int maxRows = InpNewsCalendarCount;
   if(maxRows < 1) maxRows = 1;
   if(maxRows > NEWS_MAX_ROWS) maxRows = NEWS_MAX_ROWS;

   g_newsCount = MathMin(cnt, maxRows);
   for(int i = 0; i < g_newsCount; i++)
     {
      g_newsDateTime[i] = tTimes[i];
      g_newsDay[i]       = DayNameFF(tTimes[i]);
      g_newsDate[i]      = TimeToString(tTimes[i], TIME_DATE);
      g_newsTime[i]      = TimeToString(tTimes[i], TIME_MINUTES);
      g_newsCurrency[i]  = tCur[i];
      g_newsName[i]      = tNames[i];
     }

   g_newsDataReady = true;
  }

//+------------------------------------------------------------------+
//| UpdateNewsPanelTexts()                                             |
//| RefreshNewsCalendar() natijasidagi g_news* massivlarni panel       |
//| ustidagi matn obyektlariga (news_row1_N / news_row2_N) chiqaradi.  |
//| Yangilik topilmasa yoki kalendar hali tayyor bo'lmasa, tushunarli   |
//| xabar ko'rsatiladi - hech qachon bo'sh yoki xato matn qolmaydi.    |
//+------------------------------------------------------------------+
void UpdateNewsPanelTexts()
  {
   if(!newsPanelOpen) return;
   if(panelMinimized) return;

   int newsRows = InpNewsCalendarCount;
   if(newsRows < 1) newsRows = 1;
   if(newsRows > NEWS_MAX_ROWS) newsRows = NEWS_MAX_ROWS;

   for(int i = 0; i < newsRows; i++)
     {
      string r1name    = prefix+"news_row1_"+(string)i;
      string r2name    = prefix+"news_row2_"+(string)i;
      string curName   = prefix+"news_cur_"+(string)i;
      string curBgName = prefix+"news_curbg_"+(string)i;
      string impName   = prefix+"news_impact_"+(string)i;
      if(ObjectFind(0, r1name) < 0) continue; // panel hali chizilmagan

      if(i < g_newsCount)
        {
         // --- ForexFactory saytidagi kabi: chapda qizil "impact" chizig'i + valyuta
         // --- badge'i (masalan "USD") + sana/vaqt (kichik) + yangilik nomi (katta) ---
         string line1 = StringFormat("%s, %s   %s", g_newsDay[i], g_newsDate[i], g_newsTime[i]);
         string line2 = g_newsName[i];
         if(StringLen(line2) > 40) line2 = StringSubstr(line2, 0, 37) + "...";

         ObjectSetString(0, r1name, OBJPROP_TEXT, line1);
         ObjectSetString(0, r2name, OBJPROP_TEXT, line2);
         ObjectSetString(0, curName, OBJPROP_TEXT, g_newsCurrency[i]);
         if(ObjectFind(0, curBgName) >= 0) ObjectSetInteger(0, curBgName, OBJPROP_BGCOLOR, FF_CURBG);
         if(ObjectFind(0, impName)   >= 0) ObjectSetInteger(0, impName,   OBJPROP_BGCOLOR, FF_RED);
        }
      else
        {
         string emptyMsg = g_newsDataReady
                            ? ((robotLang == 1) ? "— yangilik topilmadi —" : ((robotLang == 3) ? "— новостей нет —" : "— no news found —"))
                            : ((robotLang == 1) ? "Kalendar yuklanmoqda..." : ((robotLang == 3) ? "Календарь загружается..." : "Loading calendar..."));
         ObjectSetString(0, r1name, OBJPROP_TEXT, "");
         ObjectSetString(0, r2name, OBJPROP_TEXT, (i == 0) ? emptyMsg : "");
         ObjectSetString(0, curName, OBJPROP_TEXT, "");
         if(ObjectFind(0, curBgName) >= 0) ObjectSetInteger(0, curBgName, OBJPROP_BGCOLOR, FF_BG);
         if(ObjectFind(0, impName)   >= 0) ObjectSetInteger(0, impName,   OBJPROP_BGCOLOR, FF_BG);
        }
     }
  }

//+------------------------------------------------------------------+
//| UpdateNewsCountdownLine()                                          |
//| Panel ochiq bo'lganda HAR OnTimer tikida (50ms) chaqiriladi va     |
//| ro'yxatdagi ENG YAQIN yangilikkacha (g_newsDateTime[0]) qolgan      |
//| vaqtni "HH:MM:SS" formatida yangilaydi - bu ARZON amal, og'ir       |
//| CalendarValueHistory() so'rovi bunda ISHLATILMAYDI.                 |
//+------------------------------------------------------------------+
void UpdateNewsCountdownLine()
  {
   if(!newsPanelOpen) return;
   if(panelMinimized) return;
   if(ObjectFind(0, prefix+"news_cd_val") < 0) return;

   string val = "--:--:--";
   if(g_newsCount > 0)
     {
      datetime target = g_newsDateTime[0];
      datetime now    = TimeTradeServer();
      long diff = (long)(target - now);
      if(diff < 0) diff = 0;
      int hh = (int)(diff / 3600);
      int mm = (int)((diff % 3600) / 60);
      int ss = (int)(diff % 60);
      val = StringFormat("%02d:%02d:%02d", hh, mm, ss);
     }
   ObjectSetString(0, prefix+"news_cd_val", OBJPROP_TEXT, val);
  }

//+------------------------------------------------------------------+
//| NormalizeLot()                                                     |
//| Lot yaxlitlash va broker tekshiruvi:                                |
//|  1) Brokerning haqiqiy lot qadami (SYMBOL_VOLUME_STEP) asosida      |
//|     kiritilgan lot necha "qadam"ga to'g'ri kelishini hisoblaydi.    |
//|  2) Qadam ichidagi qoldiq (0.0 - 1.0 ulush) 0.5 dan KATTA bo'lsa    |
//|     (masalan step=0.01 da 0.016 -> 1.6 qadam -> qoldiq 0.6) YUQORI  |
//|     qadamga yaxlitlanadi (natija: 0.02).                            |
//|     Qoldiq 0.5 ga TENG yoki undan KICHIK bo'lsa (masalan 0.015 ->   |
//|     1.5 qadam -> qoldiq 0.5) PASTKI qadamga yaxlitlanadi (0.01).    |
//|  3) Natija brokerning minimal/maksimal ruxsat etilgan lotidan       |
//|     (SYMBOL_VOLUME_MIN / SYMBOL_VOLUME_MAX) tashqariga chiqmasligi  |
//|     uchun avtomatik ravishda chegaralanadi.                         |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| ParseNfpTimeInput()                                                |
//| Panel ustidagi "NFP VAQTI" edit maydoniga kiritilgan matnni        |
//| tekshiradi. Ikki xil xato turi alohida ajratiladi:                  |
//|                                                                    |
//|  1) FORMAT XATOSI (qaytadi: 1) - matn "HH:MM:SS" tuzilishiga mos    |
//|     kelmaydi (masalan harf/belgi bor, ":" lar yo'q yoki noto'g'ri   |
//|     joyda, bo'lim uzunligi noto'g'ri). Masalan "111:11:!11".        |
//|     Bunday holda funksiya FALSE natija beradi va chaqiruvchi kod    |
//|     ESKI (joriy) qiymatni O'ZGARTIRMAYDI - ya'ni maydon o'z         |
//|     holiga (oldingi to'g'ri qiymatga) qaytadi.                      |
//|                                                                    |
//|  2) DIAPAZON XATOSI (qaytadi: 2) - uchta bo'lim ham sof raqam,      |
//|     lekin son 00:00:00 - 23:59:59 oralig'idan tashqarida (masalan   |
//|     "24:00:00" yoki "23:60:00"). Bunday holda chaqiruvchi kod       |
//|     standart "15:29:55" ga qaytaradi.                                |
//|                                                                    |
//|  0 = OK - outNormalized ichiga "HH:MM:SS" (har doim 2 xonali,        |
//|      masalan "09:05:03") formatda yozib qaytariladi.                |
//+------------------------------------------------------------------+
int ParseNfpTimeInput(string text, string &outNormalized)
  {
   string parts[];
   int n = StringSplit(text, ':', parts);
   if(n != 3) return 1; // "HH:MM:SS" emas -> format xatosi

   int nums[3];
   for(int p = 0; p < 3; p++)
     {
      string s = parts[p];
      int len = StringLen(s);
      if(len == 0 || len > 2) return 1; // bo'sh yoki 2 xonadan uzun -> format xatosi

      for(int c = 0; c < len; c++)
        {
         ushort ch = StringGetCharacter(s, c);
         if(ch < '0' || ch > '9') return 1; // raqamdan boshqa belgi (masalan "!") -> format xatosi
        }
      nums[p] = (int)StringToInteger(s);
     }

   int hh = nums[0], mm = nums[1], ss = nums[2];
   if(hh < 0 || hh > 23) return 2; // diapazon xatosi (masalan 24:00:00)
   if(mm < 0 || mm > 59) return 2;
   if(ss < 0 || ss > 59) return 2;

   outNormalized = StringFormat("%02d:%02d:%02d", hh, mm, ss);
   return 0;
  }

//+------------------------------------------------------------------+
//| FormatUzTime()                                                     |
//| Vaqtni sodda, tez o'qiladigan o'zbekcha formatga o'giradi:         |
//| masalan "2026-yil 13-iyul, 18:15:48"                               |
//+------------------------------------------------------------------+
string FormatUzTime(datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   string months[12] = {"yanvar","fevral","mart","aprel","may","iyun",
                         "iyul","avgust","sentabr","oktabr","noyabr","dekabr"};
   string monthName = (dt.mon >= 1 && dt.mon <= 12) ? months[dt.mon - 1] : "";
   return StringFormat("%d-yil %d-%s, %02d:%02d:%02d",
                        dt.year, dt.day, monthName, dt.hour, dt.min, dt.sec);
  }

//+------------------------------------------------------------------+
//| UrlEncode()                                                        |
//| Matnni URL-safe (percent-encoded) formatga o'giradi.               |
//| MUHIM: emoji va boshqa maxsus belgilar bir nechta BAYTdan iborat   |
//| bo'ladi (UTF-8'da), shuning uchun matn avval to'liq UTF-8 bayt     |
//| massiviga aylantiriladi, so'ng har bir BAYT alohida %XX qilib      |
//| kodlanadi - harf-harf emas, aynan bayt-bayt. Aks holda emoji kabi  |
//| ko'p baytli belgilar buzilib, Telegram "Bad Request: strings must  |
//| be encoded in UTF-8" xatosini qaytaradi.                          |
//+------------------------------------------------------------------+
string UrlEncode(string text)
  {
   uchar bytes[];
   int rawLen = StringToCharArray(text, bytes, 0, WHOLE_ARRAY, CP_UTF8);
   int len = (rawLen > 0) ? rawLen - 1 : 0; // oxiridagi '\0' ni hisobga olmaymiz

   string result = "";
   for(int i = 0; i < len; i++)
     {
      uchar b = bytes[i];
      if((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') ||
         b == '-' || b == '_' || b == '.' || b == '~')
        {
         result += CharToString(b);
        }
      else
        {
         result += StringFormat("%%%02X", b);
        }
     }
   return result;
  }

//+------------------------------------------------------------------+
//| SendTelegramMessage()                                              |
//| Telegram Bot API orqali (HTTP POST) xabar yuboradi.                |
//|                                                                     |
//| MUHIM SOZLASH (bir martalik): MetaTrader 5 terminalida             |
//|   Tools -> Options -> Expert Advisors bo'limida                    |
//|   "Allow WebRequest for listed URL" belgilanadi, va ro'yxatga      |
//|   https://api.telegram.org qo'shiladi. Aks holda WebRequest()      |
//|   har doim xato (-1) qaytaradi.                                     |
//|                                                                     |
//| InpTelegramEnabled = false bo'lsa yoki Token/ChatID bo'sh bo'lsa,   |
//| funksiya jim chiqib ketadi (hech qanday xato bermaydi) - shuning   |
//| uchun token hali qo'yilmagan bo'lsa ham EA xavfsiz ishlayveradi.   |
//+------------------------------------------------------------------+
void SendTelegramMessage(string text)
  {
   if(!InpTelegramEnabled) return;
   if(StringLen(InpTelegramBotToken) == 0 || StringLen(InpTelegramChatID) == 0) return;

   string url  = "https://api.telegram.org/bot" + InpTelegramBotToken + "/sendMessage";
   string body = "chat_id=" + InpTelegramChatID + "&text=" + UrlEncode(text);

   char   postData[];
   int    rawLen = StringToCharArray(body, postData, 0, WHOLE_ARRAY, CP_UTF8);
   if(rawLen > 0) ArrayResize(postData, rawLen - 1); // oxiridagi '\0' ni olib tashlaymiz

   char   result[];
   string resultHeaders;
   string headers = "Content-Type: application/x-www-form-urlencoded\r\n";

   ResetLastError();
   int res = WebRequest("POST", url, headers, 5000, postData, result, resultHeaders);
   if(res == -1)
     {
      int err = GetLastError();
      Print("AzzoTrade Telegram: WebRequest xatosi (", err, "). Terminalda Tools->Options->Expert Advisors ",
            "bo'limiga https://api.telegram.org manzilini qo'shganingizni tekshiring.");
     }
   else
     {
      // --- WebRequest o'zi muvaffaqiyatli bo'lsa ham, Telegram tokeni yoki
      // --- Chat ID noto'g'ri bo'lsa, Telegram bu haqda JSON javobda xabar beradi
      // --- (masalan "Unauthorized" yoki "chat not found"). Shu javobni har doim
      // --- jurnalga chiqaramiz - shunda muammo aynan qayerdaligi ko'rinadi.
      string respText = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
      if(StringFind(respText, "\"ok\":true") < 0)
         Print("AzzoTrade Telegram: HTTP kod=", res, " | Telegram javobi: ", respText);
      else
         Print("AzzoTrade Telegram: xabar muvaffaqiyatli yuborildi.");
     }
  }

//+------------------------------------------------------------------+
//| FlushOpenBatch()                                                   |
//| To'plangan barcha "order ochildi" hodisalarini BITTA Telegram      |
//| xabariga birlashtirib yuboradi (bir zumda bir nechtasi ochilsa,   |
//| barchasi birga; ketma-ket ochilsa, har biri o'z vaqtida alohida). |
//+------------------------------------------------------------------+
void FlushOpenBatch()
  {
   int cnt = ArraySize(g_openBatchType);
   if(cnt > 0)
     {
      double totalLot = 0, totalProfit = 0;
      string details = "";
      for(int i = 0; i < cnt; i++)
        {
         totalLot    += g_openBatchLot[i];
         totalProfit += g_openBatchProfit[i];
         string sideIcon = (g_openBatchType[i] == "BUY") ? "🟩" : "🟥";
         details += StringFormat("%s %s %s lot @ %s (%s$)\n",
                       sideIcon, g_openBatchType[i], DoubleToString(g_openBatchLot[i], 2),
                       DoubleToString(g_openBatchPrice[i], _Digits),
                       DoubleToString(g_openBatchProfit[i], 2));
        }
      string statusIcon = (totalProfit >= 0) ? "📈" : "📉";
      int    spreadPts  = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);

      string msg = StringFormat(
         "🚀 ORDER OCHILDI!\n\n"
         "💱 PARA: %s\n"
         "🔢 NECHTA ORDER: %d ta\n"
         "📦 JAMI LOT: %s\n"
         "💵 HOLAT: %s USD %s\n"
         "📏 SPREAD: %d pts\n\n"
         "📋 TAFSILOTLAR:\n%s\n"
         "⏰ VAQT: %s",
         _Symbol, cnt, DoubleToString(totalLot, 2), DoubleToString(totalProfit, 2), statusIcon,
         spreadPts, details,
         FormatUzTime(TimeCurrent()));

      SendTelegramMessage(msg);
     }

   ArrayFree(g_openBatchType);
   ArrayFree(g_openBatchLot);
   ArrayFree(g_openBatchPrice);
   ArrayFree(g_openBatchProfit);
   g_openBatchActive = false;
  }

//+------------------------------------------------------------------+
//| FlushCloseBatch()                                                  |
//| To'plangan barcha "order yopildi" hodisalarini BITTA Telegram      |
//| xabariga birlashtirib yuboradi. Bitta order yopilsa - o'sha bitta  |
//| haqida; "HAMMASINI TOZALASH" bosilganda bir nechtasi bir zumda     |
//| yopilsa - barchasi birga, umumiy jami bilan.                       |
//+------------------------------------------------------------------+
void FlushCloseBatch()
  {
   int cnt = ArraySize(g_closeBatchType);
   if(cnt > 0)
     {
      double totalLot = 0, totalProfit = 0;
      int    totalPoints = 0;
      string details = "";
      for(int i = 0; i < cnt; i++)
        {
         totalLot    += g_closeBatchLot[i];
         totalProfit += g_closeBatchProfit[i];
         totalPoints += g_closeBatchPoints[i];
         string sideIcon = (g_closeBatchType[i] == "BUY") ? "🟩" : "🟥";
         string ptsIcon  = (g_closeBatchPoints[i] >= 0) ? "✅" : "❌";
         details += StringFormat("%s %s %s lot | %s ➜ %s | %s%d Pts | %s$\n",
                       sideIcon, g_closeBatchType[i], DoubleToString(g_closeBatchLot[i], 2),
                       DoubleToString(g_closeBatchOpenPrice[i], _Digits),
                       DoubleToString(g_closeBatchClosePrice[i], _Digits),
                       ptsIcon, g_closeBatchPoints[i], DoubleToString(g_closeBatchProfit[i], 2));
        }

      // --- Natijaga qarab bitta neytral emoji (stiker emojilarsiz) ---
      string boom       = (totalProfit >= 0) ? "💰" : "📉";
      string status     = (totalProfit >= 0) ? "FOYDA" : "ZARAR";
      string resultIcon  = (totalProfit >= 0) ? "✅" : "❌";

      string msg = StringFormat(
         "%s ORDER YOPILDI - %s\n\n"
         "💱 PARA: %s\n"
         "🔢 NECHTA ORDER: %d ta\n"
         "📦 JAMI LOT: %s\n"
         "🎯 JAMI POINT: %d Pts\n"
         "💰 JAMI NATIJA: %s USD %s\n\n"
         "📋 TAFSILOTLAR:\n%s\n"
         "⏰ VAQT: %s\n\n"
         "💪 HECH QACHON TASLIM BO'LMA!",
         boom, status, _Symbol, cnt, DoubleToString(totalLot, 2),
         totalPoints,
         DoubleToString(totalProfit, 2), resultIcon, details,
         FormatUzTime(TimeCurrent()));

      // --- Skrinshot funksiyasi olib tashlangan - endi har doim faqat ---
      // --- matnli Telegram xabari yuboriladi. ---
      SendTelegramMessage(msg);
     }

   ArrayFree(g_closeBatchType);
   ArrayFree(g_closeBatchLot);
   ArrayFree(g_closeBatchOpenPrice);
   ArrayFree(g_closeBatchClosePrice);
   ArrayFree(g_closeBatchProfit);
   ArrayFree(g_closeBatchPoints);
   g_closeBatchActive = false;
  }

// --- YANGI: INFO paneldagi "Order N : Lot" qatorlarini hisoblaydi.
// --- Birinchi 5 order HAR DOIM alohida-alohida qator bo'ladi (1,2,3,4,5 - har
// --- birining o'z Lot maydoni bilan). 5 dan keyingi BARCHA orderlar esa BITTA
// --- guruhga tushadi (masalan orderCount=20 bo'lsa - "6-20" degan YAGONA qator,
// --- orderCount=50 bo'lsa - "6-50" degan YAGONA qator) - shu guruhning Lot
// --- maydoniga yozilgan qiymat guruhdagi BARCHA orderlarga baravariga
// --- qo'llanadi. rowStart/rowEnd - har qatorning 0-based order diapazoni
// --- (ikkisi ham inklyuziv). Qaytadi: qatorlar soni (eng ko'pi bilan 6 ta:
// --- 1,2,3,4,5 + bitta guruh).
int BuildInfoRows(int orderCount, int &rowStart[], int &rowEnd[])
  {
   if(orderCount < 1) orderCount = 1;
   if(orderCount > AZZO_MAX_GRID_ORDERS) orderCount = AZZO_MAX_GRID_ORDERS;

   ArrayResize(rowStart, 0);
   ArrayResize(rowEnd,   0);
   int rows = 0;

   int individualCount = (orderCount < 5) ? orderCount : 5;
   for(int i = 0; i < individualCount; i++)
     {
      ArrayResize(rowStart, rows + 1);
      ArrayResize(rowEnd,   rows + 1);
      rowStart[rows] = i;
      rowEnd[rows]   = i;
      rows++;
     }

   // --- 5 dan keyingi qolgan BARCHA orderlar (bo'lsa) - BITTA yagona qatorga
   // --- yig'iladi, 5 tadan bo'lib-bo'lib guruhlanmaydi.
   if(orderCount > individualCount)
     {
      ArrayResize(rowStart, rows + 1);
      ArrayResize(rowEnd,   rows + 1);
      rowStart[rows] = individualCount;
      rowEnd[rows]   = orderCount - 1;
      rows++;
     }

   return rows;
  }

// --- YANGI: har bir order (0-based index) uchun bazaviy narxdan (Ask/Bid dan
// --- InpDistance masofada turgan 1-order narxidan) necha PUNKT uzoqlikda
// --- turishini hisoblaydi. Endi bitta umumiy InpGridStep o'rniga UCHTA alohida
// --- masofa ishlatiladi:
// ---   1) InpGridStepGroup1       - 1,2,3,4,5-orderlar O'ZARO orasidagi qadam
// ---   2) InpGridGapBetweenGroups - 5-order bilan 6-order ORASIDAGI (guruhlar
// ---                                orasidagi) QO'SHIMCHA masofa
// ---   3) InpGridStepGroup2       - 6-order dan boshlab O'ZARO orasidagi qadam
// --- Masalan Group1Step=60, Gap=250, Group2Step=60 bo'lsa:
// ---   order1=0, order2=60, order3=120, order4=180, order5=240,
// ---   order6=240+250=490, order7=490+60=550, order8=610, ...
double GetGridOffsetPoints(int orderIndex0based)
  {
   int i = orderIndex0based;
   if(i < 0) i = 0;

   if(i < 5)
      return (double)(i * g_gridStep1);

   double offsetAtOrder5 = 4.0 * g_gridStep1;
   double offsetAtOrder6 = offsetAtOrder5 + g_gridGap;
   return offsetAtOrder6 + (double)((i - 5) * g_gridStep2);
  }

double NormalizeLot(double lot)
  {
   double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0) step = 0.01;
   if(minLot <= 0) minLot = 0.01;

   if(lot < 0) lot = 0;

   double stepsRaw    = lot / step;
   double stepsFloor   = MathFloor(stepsRaw);
   double remainder    = stepsRaw - stepsFloor;        // 0.0 dan 1.0 gacha bo'lgan ulush
   double stepsRounded = (remainder > 0.5 + 0.0000001) ? stepsFloor + 1.0 : stepsFloor;

   double normalized = stepsRounded * step;

   // --- Broker chegaralariga moslashtirish ---
   if(normalized < minLot) normalized = minLot;   // masalan 0.0001 kiritilsa -> 0.01 ga tenglashtiriladi
   if(normalized > maxLot) normalized = maxLot;

   int lotDigits = (step < 0.01) ? 3 : 2;  // brokerda mayda qadam (masalan 0.001) bo'lsa aniqlik yo'qolmaydi
   return NormalizeDouble(normalized, lotDigits);
  }