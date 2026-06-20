//+------------------------------------------------------------------+
//|                                    AZZO_NFP_HACKER_PRO_EA_v11.mq5 |
//|                                           Azzo Trade Algorithmic |
//|   ENHANCED: Full Fix + Optimization + Risk Management + Hotkeys   |
//+------------------------------------------------------------------+
#property copyright   "Azzo Trade"
#property link        "https://t.me/azzo_trade"
#property version     "11.00"
#property description "Professional Trading Robot with Full Risk Management"
#property description "Shift+X Toggle | Shift+Z Kill All | Auto Lot | Trailing Stop | Break Even"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>

//+------------------------------------------------------------------+
//|  ASOSIY SAVDO SOZLAMALARI (MAIN TRADING PARAMETERS)              |
//+------------------------------------------------------------------+
input group "=== 💰 ASOSIY SAVDO SOZLAMALARI ==="
input double        InpLotSize     = 0.01;         // Boshlang'ich Lot
input int           InpTakeProfit  = 500;          // Take Profit (Pips)
input int           InpStopLoss    = 300;          // Stop Loss (Pips)
input ulong         InpMagicNumber = 777888;       // Magic Number
input string        InpOrderComment= "Azzo Pro v11"; // Order Comment

input group "=== 🛡️ RISK MANAGEMENT ==="
input bool          InpUseAutoLot  = false;        // Auto Lot System
input double        InpRiskPercent = 1.0;          // Risk % per Trade
input double        InpMaxSpread   = 30.0;         // Max Spread (Pips)
input int           InpMaxTrades   = 5;            // Max Concurrent Trades
input double        InpMaxDailyLoss= 5.0;          // Max Daily Loss %

input group "=== 📈 TRAILING STOP & BREAK EVEN ==="
input bool          InpUseTrailing = true;         // Enable Trailing Stop
input int           InpTrailingStop= 150;          // Trailing Stop (Pips)
input int           InpTrailingStep= 50;           // Trailing Step (Pips)
input bool          InpUseBreakEven= true;         // Enable Break Even
input int           InpBreakEvenTrigger = 100;     // Break Even Trigger (Pips)

input group "=== ⚡ RSI INDIKATORI SOZLAMALARI ==="
input int           InpRSIPeriod   = 14;           // RSI Period
input double        InpSellLevel   = 71.0;         // RSI Sell Level
input double        InpBuyLevel    = 30.0;         // RSI Buy Level
input ENUM_APPLIED_PRICE InpRSIPrice = PRICE_CLOSE;

input group "=== 📊 BOLLINGER BANDS SOZLAMALARI ==="
input int           InpBBPeriod    = 20;           // BB Period
input double        InpBBDeviation = 2.0;          // BB Deviation
input int           InpBBShift     = 0;            // BB Shift
input ENUM_APPLIED_PRICE InpBBPrice = PRICE_CLOSE;

input group "=== 🖥️ INTERFEYS ==="
input bool          InpShowIndicators = true;     // Show Indicators on Chart
input bool          InpShowPanel   = true;         // Show Control Panel

//--- Global Variables
CTrade         Trade;
CPositionInfo  PosInfo;
CSymbolInfo    SymInfo;
CAccountInfo   AccInfo;

bool           IsEaActive     = false;
int            HandleRSI      = INVALID_HANDLE;
int            HandleBB       = INVALID_HANDLE;
double         BufferRSI[], BufferUp[], BufferMid[], BufferLow[];
datetime       LastBarTime    = 0;
double         DailyStartBalance = 0;

//+------------------------------------------------------------------+
//|  TRADE MANAGER CLASS (SAVDO BOSHQARUV SINFI)                     |
//+------------------------------------------------------------------+
class CTradeManager
{
private:
   double         m_lastTradeProfit;
   int            m_tradesThisBar;

   double         CalculateLotSize();
   bool           CheckSpread();
   bool           CheckMaxTrades();
   bool           CheckDailyLossLimit();
   int            CountOpenTrades();

public:
                  CTradeManager() { Trade.SetExpertMagicNumber(InpMagicNumber); m_lastTradeProfit = 0; m_tradesThisBar = 0; }
   void           KillAllOrders();
   void           OpenBuy();
   void           OpenSell();
   void           ManageTrailingStop();
   void           ManageBreakEven();
   bool           IsReadyToTrade();
   double         GetLastTradeProfit() { return m_lastTradeProfit; }
};

double CTradeManager::CalculateLotSize()
{
   if(!InpUseAutoLot) return InpLotSize;
   
   SymInfo.Name(_Symbol);
   SymInfo.RefreshRates();
   
   double free_margin = AccInfo.FreeMargin();
   double tick_value = SymInfo.TickValue();
   double tick_size = SymInfo.TickSize();
   
   if(tick_value <= 0 || tick_size <= 0 || free_margin <= 0) return InpLotSize;
   
   double risk_amount = free_margin * (InpRiskPercent / 100.0);
   double lot = risk_amount / (InpStopLoss * tick_value / tick_size);
   
   double min_lot = SymInfo.LotsMin();
   double max_lot = SymInfo.LotsMax();
   double step_lot = SymInfo.LotsStep();
   
   lot = MathFloor(lot / step_lot) * step_lot;
   if(lot < min_lot) lot = min_lot;
   if(lot > max_lot) lot = max_lot;
   
   return lot;
}

bool CTradeManager::CheckSpread()
{
   SymInfo.Name(_Symbol);
   SymInfo.RefreshRates();
   double spread = SymInfo.Ask() - SymInfo.Bid();
   double spread_points = spread / _Point;
   return (spread_points <= InpMaxSpread);
}

int CTradeManager::CountOpenTrades()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PosInfo.SelectByIndex(i) && PosInfo.Symbol() == _Symbol && PosInfo.Magic() == InpMagicNumber)
         count++;
   }
   return count;
}

bool CTradeManager::CheckMaxTrades()
{
   return (CountOpenTrades() < InpMaxTrades);
}

bool CTradeManager::CheckDailyLossLimit()
{
   double current_balance = AccInfo.Balance();
   double daily_loss = DailyStartBalance - current_balance;
   double loss_percent = (daily_loss / DailyStartBalance) * 100.0;
   return (loss_percent < InpMaxDailyLoss);
}

bool CTradeManager::IsReadyToTrade()
{
   return (CheckSpread() && CheckMaxTrades() && CheckDailyLossLimit() && IsEaActive);
}

void CTradeManager::KillAllOrders()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PosInfo.SelectByIndex(i) && PosInfo.Symbol() == _Symbol && PosInfo.Magic() == InpMagicNumber)
      {
         Trade.PositionClose(PosInfo.Ticket());
      }
   }
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket > 0 && OrderGetInteger(ORDER_MAGIC) == InpMagicNumber)
         Trade.OrderDelete(ticket);
   }
   PlaySound("ok.wav");
}

void CTradeManager::OpenBuy()
{
   if(!IsReadyToTrade()) return;
   
   SymInfo.Name(_Symbol);
   SymInfo.RefreshRates();
   
   double ask = SymInfo.Ask();
   double sl = (InpStopLoss > 0) ? ask - InpStopLoss * _Point : 0;
   double tp = (InpTakeProfit > 0) ? ask + InpTakeProfit * _Point : 0;
   double lot = CalculateLotSize();
   
   if(Trade.Buy(lot, _Symbol, ask, sl, tp, InpOrderComment))
   {
      m_lastTradeProfit = InpTakeProfit * _Point;
      PlaySound("tick.wav");
   }
}

void CTradeManager::OpenSell()
{
   if(!IsReadyToTrade()) return;
   
   SymInfo.Name(_Symbol);
   SymInfo.RefreshRates();
   
   double bid = SymInfo.Bid();
   double sl = (InpStopLoss > 0) ? bid + InpStopLoss * _Point : 0;
   double tp = (InpTakeProfit > 0) ? bid - InpTakeProfit * _Point : 0;
   double lot = CalculateLotSize();
   
   if(Trade.Sell(lot, _Symbol, bid, sl, tp, InpOrderComment))
   {
      m_lastTradeProfit = InpTakeProfit * _Point;
      PlaySound("tick.wav");
   }
}

void CTradeManager::ManageTrailingStop()
{
   if(!InpUseTrailing) return;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PosInfo.SelectByIndex(i) || PosInfo.Symbol() != _Symbol || PosInfo.Magic() != InpMagicNumber)
         continue;
      
      SymInfo.Name(_Symbol);
      SymInfo.RefreshRates();
      
      if(PosInfo.PositionType() == POSITION_TYPE_BUY)
      {
         double current_sl = PosInfo.StopLoss();
         double new_sl = SymInfo.Bid() - InpTrailingStop * _Point;
         
         if(SymInfo.Bid() - PosInfo.PriceOpen() > InpTrailingStop * _Point)
         {
            if(current_sl < new_sl || current_sl == 0)
               Trade.PositionModify(PosInfo.Ticket(), new_sl, PosInfo.TakeProfit());
         }
      }
      else if(PosInfo.PositionType() == POSITION_TYPE_SELL)
      {
         double current_sl = PosInfo.StopLoss();
         double new_sl = SymInfo.Ask() + InpTrailingStop * _Point;
         
         if(PosInfo.PriceOpen() - SymInfo.Ask() > InpTrailingStop * _Point)
         {
            if(current_sl > new_sl || current_sl == 0)
               Trade.PositionModify(PosInfo.Ticket(), new_sl, PosInfo.TakeProfit());
         }
      }
   }
}

void CTradeManager::ManageBreakEven()
{
   if(!InpUseBreakEven) return;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PosInfo.SelectByIndex(i) || PosInfo.Symbol() != _Symbol || PosInfo.Magic() != InpMagicNumber)
         continue;
      
      SymInfo.Name(_Symbol);
      SymInfo.RefreshRates();
      
      if(PosInfo.PositionType() == POSITION_TYPE_BUY)
      {
         if(SymInfo.Bid() - PosInfo.PriceOpen() >= InpBreakEvenTrigger * _Point && 
            PosInfo.StopLoss() < PosInfo.PriceOpen())
            Trade.PositionModify(PosInfo.Ticket(), PosInfo.PriceOpen() + (10 * _Point), PosInfo.TakeProfit());
      }
      else if(PosInfo.PositionType() == POSITION_TYPE_SELL)
      {
         if(PosInfo.PriceOpen() - SymInfo.Ask() >= InpBreakEvenTrigger * _Point && 
            (PosInfo.StopLoss() > PosInfo.PriceOpen() || PosInfo.StopLoss() == 0))
            Trade.PositionModify(PosInfo.Ticket(), PosInfo.PriceOpen() - (10 * _Point), PosInfo.TakeProfit());
      }
   }
}

CTradeManager AzzoTrade;

//+------------------------------------------------------------------+
//|  SIGNAL ENGINE CLASS (INDIKATOR SIGNALLARI)                      |
//+------------------------------------------------------------------+
class CSignalEngine
{
private:
   double m_currentRSI;
   double m_bbUp, m_bbMid, m_bbLow;
   double m_currentPrice;
   int    m_shapeSignal;

public:
                     CSignalEngine();
   bool              InitIndicators();
   void              UpdateSignals();
   void              DrawSignalShape(int type);
   
   double            GetRSI()       { return m_currentRSI; }
   double            GetBBUp()      { return m_bbUp; }
   double            GetBBLow()     { return m_bbLow; }
   double            GetPrice()     { return m_currentPrice; }
   int               GetShapeSignal() { return m_shapeSignal; }
   
   int               CheckTradeSignal();
};

CSignalEngine::CSignalEngine()
{
   ArraySetAsSeries(BufferRSI, true);
   ArraySetAsSeries(BufferUp, true);
   ArraySetAsSeries(BufferMid, true);
   ArraySetAsSeries(BufferLow, true);
   m_shapeSignal = 0;
}

bool CSignalEngine::InitIndicators()
{
   HandleRSI = iRSI(_Symbol, PERIOD_CURRENT, InpRSIPeriod, InpRSIPrice);
   HandleBB  = iBands(_Symbol, PERIOD_CURRENT, InpBBPeriod, InpBBShift, InpBBDeviation, InpBBPrice);
   
   if(HandleRSI == INVALID_HANDLE || HandleBB == INVALID_HANDLE)
   {
      Print("❌ Indikator initialashtirilmadi!");
      return false;
   }
   return true;
}

void CSignalEngine::UpdateSignals()
{
   if(CopyBuffer(HandleRSI, 0, 0, 2, BufferRSI) <= 0) return;
   if(CopyBuffer(HandleBB, 1, 0, 2, BufferUp) <= 0) return;
   if(CopyBuffer(HandleBB, 0, 0, 2, BufferMid) <= 0) return;
   if(CopyBuffer(HandleBB, 2, 0, 2, BufferLow) <= 0) return;
   
   m_currentRSI   = BufferRSI[0];
   m_bbUp         = BufferUp[0];
   m_bbMid        = BufferMid[0];
   m_bbLow        = BufferLow[0];
   
   SymInfo.Name(_Symbol);
   SymInfo.RefreshRates();
   m_currentPrice = SymInfo.Bid();
   
   m_shapeSignal = 0;
   if(m_currentPrice <= m_bbLow)       { m_shapeSignal = 1;  DrawSignalShape(1);  }
   else if(m_currentPrice >= m_bbUp)   { m_shapeSignal = -1; DrawSignalShape(-1); }
}

void CSignalEngine::DrawSignalShape(int type)
{
   datetime timeCurrent = iTime(_Symbol, PERIOD_CURRENT, 0);
   string objName = "AZ_SHAPE_" + TimeToString(timeCurrent, TIME_DATE|TIME_MINUTES);
   
   if(ObjectFind(0, objName) >= 0) return;

   if(type == 1)
   {
      ObjectCreate(0, objName, OBJ_ARROW_BUY, 0, timeCurrent, m_bbLow - (10 * _Point));
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clrLime);
      ObjectSetInteger(0, objName, OBJPROP_WIDTH, 2);
   }
   else if(type == -1)
   {
      ObjectCreate(0, objName, OBJ_ARROW_SELL, 0, timeCurrent, m_bbUp + (10 * _Point));
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clrRed);
      ObjectSetInteger(0, objName, OBJPROP_WIDTH, 2);
   }
}

int CSignalEngine::CheckTradeSignal()
{
   UpdateSignals();
   if(m_currentRSI < InpBuyLevel && m_shapeSignal == 1)   return 1;
   if(m_currentRSI > InpSellLevel && m_shapeSignal == -1) return -1;
   return 0;
}

CSignalEngine Engine;

//+------------------------------------------------------------------+
//|  PANEL CLASS (INTERFEYS PANELI)                                  |
//+------------------------------------------------------------------+
class CAzzoPanel
{
private:
   string m_prefix;
   int    m_x, m_y;
   
   void   CreateLabel(string name, string text, int x_ofs, int y_ofs, color clr, int size = 11, string font = "Consolas");
   void   CreateRect(string name, int x_ofs, int y_ofs, int w, int h, color bg_clr, color border_clr);
   void   CreateButton(string name, string text, int x_ofs, int y_ofs, int w, int h, color bg, color txt_clr);

public:
                     CAzzoPanel();
                    ~CAzzoPanel();
   void              DrawPanel();
   void              UpdateDynamicData(double rsi, double price, double bb_up, double bb_low, int shape, int trades);
   void              UpdateStatusText();
   void              ClearPanel();
};

CAzzoPanel::CAzzoPanel()
{
   m_prefix = "AZ_PNL_";
   m_x = 25; 
   m_y = 35;
}

CAzzoPanel::~CAzzoPanel()
{
   ClearPanel();
}

void CAzzoPanel::ClearPanel()
{
   int total = ObjectsTotal(0, -1, -1);
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(0, i, -1, -1);
      if(StringFind(name, m_prefix) >= 0) 
         ObjectDelete(0, name);
   }
   ChartRedraw();
}

void CAzzoPanel::DrawPanel()
{
   CreateRect("BG", 0, 0, 320, 400, C'10,10,10', clrLime);
   CreateRect("HDR", 0, 0, 320, 50, C'15,25,15', clrLime);
   
   CreateLabel("TITLE", "🤖 AZZO NFP PRO v11", 15, 12, clrLime, 14, "Impact");
   
   CreateLabel("T_PARA", "PARA:", 15, 70, clrLime, 11);
   CreateLabel("V_PARA", _Symbol, 160, 70, clrWhite, 12, "Arial Bold");
   
   CreateLabel("T_LOT", "LOT:", 15, 100, clrLime, 11);
   CreateLabel("V_LOT", DoubleToString(InpLotSize, 3), 160, 100, clrWhite, 11);
   
   CreateLabel("T_RSI", "RSI:", 15, 130, clrLime, 11);
   CreateLabel("V_RSI", "-- ", 160, 130, clrWhite, 11);
   
   CreateLabel("T_BB", "BB STATUS:", 15, 160, clrLime, 11);
   CreateLabel("V_BB", "NORMAL", 160, 160, clrWhite, 11);
   
   CreateLabel("T_SIGNAL", "SIGNAL:", 15, 190, clrLime, 11);
   CreateLabel("V_SIGNAL", "WAIT", 160, 190, clrWhite, 11);
   
   CreateLabel("T_TRADES", "OPEN TRADES:", 15, 220, clrLime, 11);
   CreateLabel("V_TRADES", "0", 160, 220, clrWhite, 11);
   
   CreateLabel("T_STAT", "STATUS:", 15, 250, clrLime, 11);
   CreateLabel("V_STAT", "🔴 OFF", 160, 250, clrRed, 12, "Arial Bold");
   
   CreateButton("BTN_KILL", "[ BARCHASINI QIRISH - SHIFT+Z ]", 15, 290, 290, 32, C'60,10,10', clrWhite);
   
   CreateLabel("HOTKEYS", "✓ TOGGLE: SHIFT+X  |  ✓ KILL: SHIFT+Z", 20, 365, clrYellow, 8, "Arial");
}

void CAzzoPanel::UpdateDynamicData(double rsi, double price, double bb_up, double bb_low, int shape, int trades)
{
   string rsi_txt = DoubleToString(rsi, 1);
   color rsi_clr = clrWhite;
   if(rsi < InpBuyLevel)       { rsi_txt += " BUY"; rsi_clr = clrLime; }
   else if(rsi > InpSellLevel) { rsi_txt += " SELL"; rsi_clr = clrRed; }
   
   string bb_txt = "NORMAL";
   color bb_clr = clrWhite;
   if(price <= bb_low)       { bb_txt = "BOTTOM 📉"; bb_clr = clrLime; }
   else if(price >= bb_up)   { bb_txt = "TOP 📈"; bb_clr = clrRed; }
   
   string sig_txt = "WAIT";
   color sig_clr = clrWhite;
   if(shape == 1)       { sig_txt = "BUY 🟢"; sig_clr = clrLime; }
   else if(shape == -1) { sig_txt = "SELL 🔴"; sig_clr = clrRed; }
   
   ObjectSetString(0, m_prefix+"V_RSI", OBJPROP_TEXT, rsi_txt);
   ObjectSetInteger(0, m_prefix+"V_RSI", OBJPROP_COLOR, rsi_clr);
   
   ObjectSetString(0, m_prefix+"V_BB", OBJPROP_TEXT, bb_txt);
   ObjectSetInteger(0, m_prefix+"V_BB", OBJPROP_COLOR, bb_clr);

   ObjectSetString(0, m_prefix+"V_SIGNAL", OBJPROP_TEXT, sig_txt);
   ObjectSetInteger(0, m_prefix+"V_SIGNAL", OBJPROP_COLOR, sig_clr);
   
   ObjectSetString(0, m_prefix+"V_TRADES", OBJPROP_TEXT, IntegerToString(trades));
   ObjectSetInteger(0, m_prefix+"V_TRADES", OBJPROP_COLOR, (trades > 0) ? clrYellow : clrWhite);
}

void CAzzoPanel::UpdateStatusText()
{
   if(IsEaActive) {
      ObjectSetString(0, m_prefix+"V_STAT", OBJPROP_TEXT, "🟢 ON");
      ObjectSetInteger(0, m_prefix+"V_STAT", OBJPROP_COLOR, clrLime);
   } else {
      ObjectSetString(0, m_prefix+"V_STAT", OBJPROP_TEXT, "🔴 OFF");
      ObjectSetInteger(0, m_prefix+"V_STAT", OBJPROP_COLOR, clrRed);
   }
}

void CAzzoPanel::CreateLabel(string name, string text, int x_ofs, int y_ofs, color clr, int size, string font)
{
   string obj = m_prefix + name;
   if(ObjectFind(0, obj) < 0) ObjectCreate(0, obj, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, obj, OBJPROP_XDISTANCE, m_x + x_ofs);
   ObjectSetInteger(0, obj, OBJPROP_YDISTANCE, m_y + y_ofs);
   ObjectSetString(0, obj, OBJPROP_TEXT, text);
   ObjectSetInteger(0, obj, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, obj, OBJPROP_FONTSIZE, size);
   ObjectSetString(0, obj, OBJPROP_FONT, font);
   ObjectSetInteger(0, obj, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, obj, OBJPROP_HIDDEN, true);
}

void CAzzoPanel::CreateRect(string name, int x_ofs, int y_ofs, int w, int h, color bg_clr, color border_clr)
{
   string obj = m_prefix + name;
   if(ObjectFind(0, obj) < 0) ObjectCreate(0, obj, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, obj, OBJPROP_XDISTANCE, m_x + x_ofs);
   ObjectSetInteger(0, obj, OBJPROP_YDISTANCE, m_y + y_ofs);
   ObjectSetInteger(0, obj, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, obj, OBJPROP_YSIZE, h);
   ObjectSetInteger(0, obj, OBJPROP_BGCOLOR, bg_clr);
   ObjectSetInteger(0, obj, OBJPROP_COLOR, border_clr);
   ObjectSetInteger(0, obj, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, obj, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, obj, OBJPROP_ZORDER, -1);
}

void CAzzoPanel::CreateButton(string name, string text, int x_ofs, int y_ofs, int w, int h, color bg, color txt_clr)
{
   string obj = m_prefix + name;
   if(ObjectFind(0, obj) < 0) ObjectCreate(0, obj, OBJ_BUTTON, 0, 0, 0);
   ObjectSetInteger(0, obj, OBJPROP_XDISTANCE, m_x + x_ofs);
   ObjectSetInteger(0, obj, OBJPROP_YDISTANCE, m_y + y_ofs);
   ObjectSetInteger(0, obj, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, obj, OBJPROP_YSIZE, h);
   ObjectSetString(0, obj, OBJPROP_TEXT, text);
   ObjectSetInteger(0, obj, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(0, obj, OBJPROP_COLOR, txt_clr);
   ObjectSetInteger(0, obj, OBJPROP_FONTSIZE, 10);
   ObjectSetString(0, obj, OBJPROP_FONT, "Arial Bold");
   ObjectSetInteger(0, obj, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, obj, OBJPROP_HIDDEN, true);
}

CAzzoPanel Panel;

//+------------------------------------------------------------------+
//|  MAIN MT5 FUNCTIONS (INIT, DEINIT, TICK, CHARTEVENT)             |
//+------------------------------------------------------------------+

int OnInit()
{
   Print("✅ AZZO NFP PRO v11 - START");
   
   if(!Engine.InitIndicators()) 
   {
      Print("❌ Indikator initialization failed!");
      return INIT_FAILED;
   }
   
   if(InpShowIndicators)
   {
      ChartIndicatorAdd(0, 0, HandleBB);
      int subwindow = (int)ChartGetInteger(0, CHART_WINDOWS_TOTAL);
      if(subwindow > 0)
         ChartIndicatorAdd(0, subwindow, HandleRSI);
   }
   
   if(InpShowPanel)
   {
      Panel.DrawPanel();
      Panel.UpdateStatusText();
   }
   
   DailyStartBalance = AccInfo.Balance();
   IsEaActive = false;
   LastBarTime = 0;
   
   ChartRedraw();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(InpShowPanel) Panel.ClearPanel();
   
   if(HandleRSI != INVALID_HANDLE) IndicatorRelease(HandleRSI);
   if(HandleBB != INVALID_HANDLE)  IndicatorRelease(HandleBB);
   
   Print("🛑 EA Stopped - Reason: " + IntegerToString(reason));
   ChartRedraw();
}

void OnTick()
{
   Engine.UpdateSignals();
   
   if(InpShowPanel)
   {
      int openTrades = 0;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(PosInfo.SelectByIndex(i) && PosInfo.Symbol() == _Symbol && PosInfo.Magic() == InpMagicNumber)
            openTrades++;
      }
      Panel.UpdateDynamicData(Engine.GetRSI(), Engine.GetPrice(), Engine.GetBBUp(), Engine.GetBBLow(), Engine.GetShapeSignal(), openTrades);
   }
   
   // ✅ TUZATILDI: Faqat EA faol bo'lganda trading logic ishlasin
   AzzoTrade.ManageTrailingStop();
   AzzoTrade.ManageBreakEven();
   
   if(!IsEaActive) return; // EA faol bo'lmasa chikish
   
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime == LastBarTime) return; 
   
   int signal = Engine.CheckTradeSignal();
   if(signal == 1)
   {
      AzzoTrade.OpenBuy();
      LastBarTime = currentBarTime;
   }
   else if(signal == -1)
   {
      AzzoTrade.OpenSell();
      LastBarTime = currentBarTime;
   }
}

void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_KEYDOWN)
   {
      bool isShiftPressed = (TerminalInfoInteger(TERMINAL_KEYSTATE_SHIFT) < 0);
      if(isShiftPressed)
      {
         if(lparam == 88) // SHIFT + X = TOGGLE
         {
            IsEaActive = !IsEaActive;
            if(InpShowPanel) Panel.UpdateStatusText();
            PlaySound("ok.wav");
            Print(IsEaActive ? "✅ EA FAOL" : "❌ EA O'CHIQ");
            ChartRedraw();
         }
         if(lparam == 90) // SHIFT + Z = KILL ALL
         {
            AzzoTrade.KillAllOrders();
            Print("🔥 Barcha orderlar o'chirildi!");
            ChartRedraw();
         }
      }
   }
   
   if(id == CHARTEVENT_OBJECT_CLICK)
   {
      if(sparam == "AZ_PNL_BTN_KILL")
      {
         ObjectSetInteger(0, "AZ_PNL_BTN_KILL", OBJPROP_STATE, false);
         AzzoTrade.KillAllOrders();
      }
   }
}
//+------------------------------------------------------------------+
//|                          END OF AZZO NFP PRO v11                  |
//+------------------------------------------------------------------+