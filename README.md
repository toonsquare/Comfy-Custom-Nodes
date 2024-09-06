# ComfyUI Custom Nodes 모음

---
## 노드 추가
- 필요한 노드가 있다면 기본 템플릿(PrintHelloWorld)을 복사해서 수정
- 내부 로직은 nodes.py에 코드 추가 후, __init_.py 내에 NODE_CLASS_MAPPINGS에 추가
- 노드 UI는 web 디렉토리내에서 js파일 추가 필요함(optional)
---

## 노드 목록
- Print Hello World
  - 노드 기본 템플릿
  - 자세한 사항은 https://github.com/Suzie1/ComfyUI_Guide_To_Making_Custom_Nodes/wiki 참고

- DeepL Translate
  - deepl api를 통해서 입력한 텍스트를 영어로 번역
  
- Image To GrayScale
  - 이미지를 흑백 이미지로 변경
    
- Wallpaper Prompt Generator
  - 삼성 월페이퍼 프로젝트 전용
  - 입력한 단어를 이용해 월페이퍼 생성을 위한 프롬프트를 gpt를 통해 생성

- Prompt Modifier
  - 입력한 프롬프트와 수정 지시사항을 기반으로 프롬프트를 수정하여 생성

- Color Reference
  - 단색 이미지 생성

- Image To ColorPalette
  - 이미지 내 색상 빈도를 기준으로 팔레트 생성

- Get Result Anything
  - 이미지 외의 값들을 api를 통해 내보내기 위한 노드
---