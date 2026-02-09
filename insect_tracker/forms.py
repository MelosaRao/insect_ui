from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField
from wtforms.validators import DataRequired, ValidationError




class UploadTrapImage(FlaskForm):
    picture = FileField("Images must follow the upload guideline", validators=[DataRequired(message="No image uploaded."), FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])
    submit = SubmitField('Upload')