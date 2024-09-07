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
from django.views.decorators.http import require_http_methods

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
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)

            # spleeter를 사용하여 파일 분리
            try:
                completed_process = subprocess.run(
                    ['python', '-m', 'spleeter', 'separate', '-o', fs.location, file_path],
                    check=True,
                    text=True,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE
                )
                print("분리 성공:", completed_process.stdout)
                
                # 분리된 파일들이 저장되는 폴더 경로
                spleeter_output_path = os.path.join(fs.location, os.path.splitext(file.name)[0])
                # 배경음악은 삭제
                file_path = os.path.join(spleeter_output_path, 'accompaniment.wav')
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print("불필요한 파일 삭제 완료")
                else:
                    print("해당 파일이 존재하지 않습니다.")

                number_of_people = request.session.get('number_of_people', '2')

                if number_of_people == '1':
                    # 화자 분리를 생략하고 바로 보이스 클론 페이지로 리디렉션
                    vocal_path = os.path.join(spleeter_output_path, 'vocals.wav')
                    speaker_path = os.path.join(spleeter_output_path, 'speaker_SPEAKER_00.wav')
                    os.rename(vocal_path, speaker_path)
                    request.session['speaker_file'] = speaker_path
                    return redirect('query_view')
                else:
                    # pyannote.audio 파이프라인 로드
                    diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token="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"
                    )

                    # 분리된 트랙들에 대한 화자 분할 실행
                    for track in os.listdir(spleeter_output_path):
                        track_path = os.path.join(spleeter_output_path, track)
                        # 오디오 파일을 불러옵니다.
                        audio_data, sample_rate = sf.read(track_path)
                        # 화자 분할을 실행합니다.
                        diarization = diarization_pipeline({'audio': track_path})

                        speakers_audio = {}
                        # 화자별로 음성 데이터를 분리합니다.
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            start_sample = int(turn.start * sample_rate)
                            end_sample = int(turn.end * sample_rate)
                            speaker_audio = audio_data[start_sample:end_sample]
                            # 화자별로 음성 데이터를 저장합니다.
                            if speaker in speakers_audio:
                                speakers_audio[speaker].append(speaker_audio)
                            else:
                                speakers_audio[speaker] = [speaker_audio]

                        # 화자별로 오디오 파일을 저장합니다.
                        for speaker, audio in speakers_audio.items():
                            # 모든 발화를 하나의 오디오 트랙으로 연결합니다.
                            speaker_audio = np.concatenate(audio, axis=0)
                            # 오디오 파일로 저장합니다.
                            speaker_output_path = os.path.join(spleeter_output_path, f'speaker_{speaker}.wav')
                            sf.write(speaker_output_path, speaker_audio, sample_rate)

                            # 파일 크기 확인
                            print(f"파일 {speaker_output_path} 크기: {os.path.getsize(speaker_output_path)} 바이트")

            except subprocess.CalledProcessError as e:
                # 오류 메시지를 컨텍스트에 추가합니다.
                context['error_message'] = e.stderr
                print(f"분리 실패: {e.stderr}")

            # 처리 후 form 초기화
            context['form'] = UploadForm()
            return redirect('select_speaker')  # 음성 파일 선택 페이지로 리디렉션

    return render(request, 'upload.html', context)

def select_speaker(request):
    speakers = [
        {'id': 0, 'file': f"{settings.MEDIA_URL}interview/speaker_SPEAKER_00.wav"},
        {'id': 1, 'file': f"{settings.MEDIA_URL}interview/speaker_SPEAKER_01.wav"},
        {'id': 2, 'file': f"{settings.MEDIA_URL}interview/speaker_SPEAKER_02.wav"}
    ]

    for speaker in speakers:
        file_path = os.path.join(settings.MEDIA_ROOT, 'interview', f'speaker_SPEAKER_0{speaker["id"]}.wav')
        if os.path.exists(file_path):
            print(f"파일 {file_path} 존재함")
            print(f"파일 {file_path} 크기: {os.path.getsize(file_path)} 바이트")
        else:
            print(f"파일 {file_path} 존재하지 않음")

    if request.method == 'POST':
        selected_speaker = request.POST.get('selected_speaker')
        request.session['selected_speaker'] = selected_speaker
        return redirect('query_view')

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
        # 초기화 세션 확인
        if 'past_messages' not in request.session:
            request.session['past_messages'] = []

        if 'file' in request.FILES:
            file_content = handle_uploaded_file(request.FILES['file'])
            # 파일 내용을 ChatGPT에게 시스템 메시지로 전달하여 세션을 초기화
            system_message = (
                "You are a helpful assistant. The user has provided a conversation file. "
                "Use this information to help the user with any questions or further discussion. "
                f"The conversation content is as follows:\n\n{file_content}\n\n"
                "When answering, please consider this context and provide clear, concise, and helpful responses."
            )
            request.session['past_messages'].append({"role": "system", "content": system_message})

            # 파일을 읽은 후 사용자에게 자동으로 메시지 전송
            initial_response = "보내주신 대화내용을 모두 읽었어요. 이제 대화를 시작할까요?"
            request.session['past_messages'].append({"role": "assistant", "content": initial_response})

            return JsonResponse({'response': initial_response})

        prompt = request.POST.get('prompt', '')
        
        if prompt:
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

            return JsonResponse({'response': answer})

    return render(request, 'chat.html')



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
