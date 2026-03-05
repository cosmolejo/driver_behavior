@echo off
setlocal enabledelayedexpansion

echo Iniciando proceso de compresion masiva...
echo Buscando en la carpeta dmd y sus subcarpetas...
echo.

REM Recorrer recursivamente la carpeta dmd buscando .mp4
for /r "dmd" %%F in (*.mp4) do (
    set "filename=%%~nF"

    REM Comprobar que el archivo no termina ya en _240 (para evitar bucles infinitos si lo pausas y lo vuelves a iniciar)
    if /I not "!filename:~-4!"=="_240" (

        set "output_file=%%~dpnF_240%%~xF"

        if not exist "!output_file!" (
            echo [PROCESANDO] %%~nxF
            ffmpeg -y -i "%%F" -c:v libx265 -crf 28 -preset fast -vf scale=240:240 -r 30 -an "!output_file!"
        ) else (
            echo [SALTANDO] %%~nxF - Ya fue comprimido previamente.
        )
    )

)

echo.
echo =========================================
echo Proceso finalizado.
echo =========================================
pause