@echo off
setlocal enabledelayedexpansion

echo Iniciando proceso de compresion y limpieza a 240p...
echo Buscando en la carpeta dmd y sus subcarpetas...
echo.

for /r "dmd" %%F in (*.mp4) do (
set "filename=%%~nF"

if /I not "!filename:~-4!"=="_240" (

    echo !filename! | findstr /I "_mosaic" >nul
    if !errorlevel! neq 0 (

        set "output_file=%%~dpnF_240%%~xF"

        if not exist "!output_file!" (
            echo [PROCESANDO] %%~nxF
            ffmpeg -y -i "%%F" -c:v libx265 -crf 28 -preset fast -vf scale=-2:240 -r 30 -an "!output_file!"

            if !errorlevel! equ 0 (
                if exist "!output_file!" (
                    echo [ELIMINANDO ORIGINAL] %%~nxF
                    del "%%F"
                )
            ) else (
                echo [ERROR] Fallo la compresion de %%~nxF. No se borrara el original.
            )
        ) else (
            echo [SALTANDO] %%~nxF - Ya fue comprimido.
        )
    ) else (
        echo [PROTEGIDO] %%~nxF - Contiene la palabra mosaic.
    )
)

)

echo.
echo =========================================
echo Proceso finalizado.
echo =========================================
pause