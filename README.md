# HTR Local para Quadro de Produção

Pipeline local para corrigir perspectiva, segmentar uma grade fixa, reconhecer escrita manual com TrOCR, revisar resultados e exportar CSV/XLSX/JSON. Nenhuma API externa é utilizada.

## Segurança e operação offline

- `run.py` força `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1`.
- O modelo é carregado com `local_files_only=True`.
- A interface escuta exclusivamente `127.0.0.1:8000`.
- Se o checkpoint não existir, a inferência falha de forma explícita. Não existe fallback cloud.
- `scripts/cache_model.py` é a única etapa que acessa a internet e deve ser executada uma vez em ambiente autorizado. Em rede isolada, copie a pasta resultante por mídia controlada.

## Instalação

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\cache_model.py
python run.py
```

Acesse `http://127.0.0.1:8000`. Para instalação CPU, use `requirements-cpu.txt`. Para CUDA, instale a distribuição PyTorch compatível com o driver local antes das demais dependências.

## Configuração do quadro

Edite `config/board_template.json`. A imagem retificada é normalizada para `canonical_size`; a primeira linha é considerada cabeçalho e as demais são divididas uniformemente conforme `rows` e `fields`. O método é adequado quando o quadro físico mantém a mesma geometria. Fotos inclinadas são corrigidas pela maior borda quadrilateral detectada.

Para produção, calibre uma foto representativa, ajuste `canonical_size`, `rows`, campos e `crop_margin`, e valide `grid_quality`. O endpoint aceita atualmente detecção automática; o módulo `rectify` já aceita quatro cantos explícitos para a futura tela de calibração visual.

## Treino incremental

Revise no mínimo 20 células e use a tela **Treino local**, ou execute:

```powershell
python scripts\train.py --epochs 3
```

O treino parte de `data/model_checkpoints/base`, grava uma nova versão sem sobrescrever a ativa e registra CER/WER em `data/metrics_log.csv` e no SQLite. A promoção deve ocorrer somente após comparar as métricas em um conjunto de validação estável. Recomenda-se iniciar com centenas de células corrigidas e manter amostras de todos os operadores.

### Importar material fotográfico

```powershell
python -m scripts.import_material --source material
```

Cada foto é corrigida, segmentada e reconhecida. Todos os resultados de baixa confiança ficam disponíveis na interface para confirmação. Sem checkpoint, use `--stub` apenas para preparar os recortes; esses recortes não são usados no treino até serem revisados por uma pessoa.

A importação é retomável: arquivos já registrados pelo mesmo nome são ignorados. Acompanhe `data/material_import.log` quando executada em segundo plano.

## Testes e modo de desenvolvimento

```powershell
pytest -q
$env:HTR_ALLOW_STUB='1'; python run.py
```

`HTR_ALLOW_STUB=1` permite testar ingestão, segmentação, revisão e exportação sem pesos; sempre retorna texto vazio e confiança zero. Não chama qualquer serviço externo e não deve ser usado para avaliar reconhecimento.

## Estrutura

- `htr_local/preprocessing.py`: perspectiva, CLAHE, grade e recortes.
- `htr_local/recognizer.py`: inferência TrOCR estritamente local.
- `htr_local/database.py`: histórico e correções SQLite.
- `htr_local/service.py`: orquestração do pipeline.
- `htr_local/web.py`: API/UI local.
- `scripts/train.py`: fine-tuning versionado.
- `data/`: imagens, células, banco, exportações e checkpoints.

## Limites atuais

Este MVP assume células uniformes e quadro fixo. Antes do uso industrial, ainda é necessário adicionar uma tela de ajuste manual dos quatro cantos e executar um conjunto de testes com fotografias reais do quadro.
