from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .forms import UploadForm, NumberOfPeopleForm
from pyannote.audio import Pipeline
import os
import soundfile as sf
import numpy as np
import openai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from elevenlabs.client import ElevenLabs, VoiceSettings
import subprocess
import logging
import base64
import json 
from django.views.decorators.http import require_http_methods
from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect

logging.basicConfig(level=logging.INFO)
openai.api_key = 'sk-D22-NEc2cS09LI3f-qiYtTXkHvICwA4eZ9Jpl-eK1wT3BlbkFJaSUR8wTiTQGQG6eAM4A7lu_e3vTuun8sL-Zp6lDUMA'
elevenlabs_api_key = '866f355f0c138130ff9afa77e40041bf'

def select_number_of_people(request):
    if request.method == 'POST':
        form = NumberOfPeopleForm(request.POST)
        if form.is_valid():
            number_of_people = form.cleaned_data['number_of_people']
            request.session['number_of_people'] = number_of_people
            return redirect('voice_separation')
    else:
        form = NumberOfPeopleForm()
    return render(request, 'select_number_of_people.html', {'form': form})

def voice_separation(request):
    context = {'form': UploadForm()}  # 초기 컨텍스트 설정

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                name = request.POST.get('name')  # 입력된 이름을 가져옴
                file = request.FILES['file']
                
                fs = FileSystemStorage()

                # 이름 기반으로 파일 저장 (name.wav 형식)
                filename = f"{name}.wav"
                file_path = fs.save(filename, file)

                full_file_path = fs.path(file_path)  # 실제 파일 경로 가져오기

                # pyannote.audio 파이프라인 로드
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token="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"
                )

                # 오디오 파일을 pyannote.audio로 처리
                audio_data, sample_rate = sf.read(full_file_path)
                diarization = diarization_pipeline({'audio': full_file_path})

                speakers_audio = {}
                speaker_files = []  # 추가: 각 화자의 파일명을 저장할 리스트

                # 화자별로 음성 데이터를 분리합니다.
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_sample = int(turn.start * sample_rate)
                    end_sample = int(turn.end * sample_rate)
                    speaker_audio = audio_data[start_sample:end_sample]

                    if speaker in speakers_audio:
                        speakers_audio[speaker].append(speaker_audio)
                    else:
                        speakers_audio[speaker] = [speaker_audio]

                # 화자별로 오디오 파일을 저장합니다.
                for speaker, audio in speakers_audio.items():
                    speaker_audio = np.concatenate(audio, axis=0)
                    speaker_output_path = os.path.join(
                        fs.location, f'{name}_speaker_{speaker}.wav'  # 이름 기반 파일명으로 저장
                    )
                    sf.write(speaker_output_path, speaker_audio, sample_rate)

                    # 파일 크기 확인
                    logging.info(f"파일 {speaker_output_path} 크기: {os.path.getsize(speaker_output_path)} 바이트")

                    # 각 화자의 파일 경로를 리스트에 추가
                    speaker_files.append(f'{name}_speaker_{speaker}.wav')

                # 세션에 화자 수와 파일명 저장
                request.session['number_of_speakers'] = len(speakers_audio)
                request.session['speaker_files'] = speaker_files  # 추가: 파일명 리스트를 세션에 저장
                
                # 화자 선택 페이지로 리디렉션
                return redirect('select_speaker')

            except Exception as e:
                # 오류 발생 시 로그 출력 및 오류 메시지 컨텍스트에 추가
                logging.error(f"화자 분리 실패: {str(e)}")
                context['error_message'] = str(e)

            # 처리 후 form 초기화
            context['form'] = UploadForm()

    return render(request, 'upload.html', context)

def select_speaker(request):
    # 세션에서 화자 수 및 화자 파일명 리스트 가져오기
    number_of_speakers = request.session.get('number_of_speakers', 0)
    speaker_files = request.session.get('speaker_files', [])  # 세션에서 파일명 리스트 가져오기
    
    speakers = []
    for i in range(number_of_speakers):
        # 저장된 파일 경로를 세션에서 불러와 사용
        speakers.append({
            'id': i,
            'file': f"{settings.MEDIA_URL}{speaker_files[i]}"  # 세션에서 저장한 파일명 사용
        })

    if request.method == 'POST':
        selected_speaker = request.POST.get('selected_speaker')
        request.session['selected_speaker'] = selected_speaker
        return HttpResponseRedirect('http://localhost:3000/chat')

    return render(request, 'select_speaker.html', {'speakers': speakers})


# ElevenLabs 클라이언트 초기화
client = ElevenLabs(api_key=elevenlabs_api_key)
 
def get_completion(prompt, past_messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=past_messages + [{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    answer = response['choices'][0]['message']['content']
    return answer

def text_to_speech(text: str) -> bytes:
    try:
        response = client.text_to_speech.convert(
            voice_id="HIfeW6Vd8E8LZ3HAkZk8",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5, similarity_boost=0.8, style=0.0, use_speaker_boost=True
            ),
        )
        audio_data = b"".join(chunk for chunk in response)
        return audio_data
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return None

def handle_uploaded_file(f):
    content = f.read().decode('utf-8')
    return content

@csrf_exempt
def query_view(request):
    if request.method == 'POST':
        try:
            # JSON 데이터 파싱
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)

            prompt = body_data.get('prompt', '')

            if not prompt:
                return JsonResponse({'error': 'No prompt provided'}, status=400)

            # 초기화 세션 확인
            if 'past_messages' not in request.session:
                request.session['past_messages'] = []

            # 사용자의 질문을 추가
            request.session['past_messages'].append({"role": "user", "content": prompt})

            # ChatGPT API 호출
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=request.session['past_messages'],
                max_tokens=1024
            )

            answer = response['choices'][0]['message']['content']

            # 응답을 저장
            request.session['past_messages'].append({"role": "assistant", "content": answer})
            request.session.modified = True

            # text_to_speech 함수 호출하여 음성 데이터를 생성
            audio_data = text_to_speech(answer)
            if audio_data is None:
                return JsonResponse({'error': 'Text-to-Speech conversion failed'}, status=500)

            # 음성 데이터를 base64로 인코딩
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            # JSON 응답에 텍스트와 음성 데이터를 함께 반환
            return JsonResponse({
                'response': answer,
                'audio_base64': audio_base64  # base64 인코딩된 음성 파일
            })

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    # POST 요청이 아닐 경우
    return JsonResponse({'error': 'Invalid request method'}, status=400)




@require_http_methods(["POST"])
def summarize_text(request):
    data = request.POST.get('text', '')
    if not data:
        return JsonResponse({'error': 'No text provided'}, status=400)

    openai.api_key = os.getenv("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")

    response = openai.Completion.create(
      engine="davinci", 
      prompt=f"요약: {data}\n\n###\n\n",
      temperature=0.5,
      max_tokens=150,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )

    summarized_text = response.choices[0].text.strip()

    return JsonResponse({'original': data, 'summary': summarized_text})

def home(request):
    return render(request, 'home.html')

def choose(request):
    return render(request, 'choose.html')

def make(request):
    return render(request, 'make.html')

def choose_ch1(request):
    return render(request, 'choose_ch1.html')

def chat(request):
    return render(request, 'chat.html')

def make_ch1(request):
    return render(request, 'make_ch1.html')
