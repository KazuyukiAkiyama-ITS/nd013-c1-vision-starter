@ECHO OFF
REM 結合バッチファイル：div2.exe により生成

ECHO 分割されたファイルを復元します

IF NOT EXIST "animation.001"  GOTO ERR1
IF NOT EXIST "animation.002"  GOTO ERR1
IF NOT EXIST "animation.003"  GOTO ERR1
IF NOT EXIST "animation.004"  GOTO ERR1
IF NOT EXIST "animation.005"  GOTO ERR1
IF NOT EXIST "animation.006"  GOTO ERR1
IF NOT EXIST "animation.007"  GOTO ERR1
IF NOT EXIST "animation.008"  GOTO ERR1
IF NOT EXIST "animation.009"  GOTO ERR1
IF NOT EXIST "animation.010"  GOTO ERR1
IF     EXIST "animation.gif"  GOTO ERR2

COPY "animation.001" "animation.gif"
IF ERRORLEVEL 1 GOTO ERR3
COPY /B "animation.gif"+"animation.002"+"animation.003"+"animation.004"+"animation.005"
IF ERRORLEVEL 1 GOTO ERR3
COPY /B "animation.gif"+"animation.006"+"animation.007"+"animation.008"+"animation.009"
IF ERRORLEVEL 1 GOTO ERR3
COPY /B "animation.gif"+"animation.010"
IF ERRORLEVEL 1 GOTO ERR3

DEL "animation.001"
DEL "animation.002"
DEL "animation.003"
DEL "animation.004"
DEL "animation.005"
DEL "animation.006"
DEL "animation.007"
DEL "animation.008"
DEL "animation.009"
DEL "animation.010"


REN "animation.gif" "animation.gif"

ECHO このバッチファイルも消してください
GOTO END

:ERR1
ECHO ファイルが一部存在しません
ECHO 処理を中止します
GOTO END

:ERR2
ECHO 復元先のファイルが既に存在しています
ECHO 処理を中止します
GOTO END

:ERR3
ECHO 復元に失敗しました
ECHO 処理を中止します
GOTO END

:END
ECHO 終了するには、何かキーを押してください
PAUSE >NUL

