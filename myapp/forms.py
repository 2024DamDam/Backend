from django import forms

class UploadForm(forms.Form):
    file = forms.FileField(label='파일을 선택하세요')

class NumberOfPeopleForm(forms.Form):
    number_of_people = forms.ChoiceField(
        choices=[('1', '1명'), ('2', '2명 이상')],
        widget=forms.RadioSelect,
        label="사람 인원 수를 선택하시오"
    )
