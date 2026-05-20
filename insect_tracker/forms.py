from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField
from wtforms.validators import DataRequired, ValidationError
from wtforms import StringField
from wtforms.validators import Optional
from wtforms.fields import DateField


class UploadTrapImage(FlaskForm):
    sample_id = StringField('Sample ID', validators=[Optional()])
    side_or_trapnum = StringField('Side / Trap Number', validators=[Optional()])
    watershed = StringField('Watershed', validators=[Optional()])
    date = DateField('Date', format='%Y-%m-%d', validators=[Optional()])
    picture = FileField("Images must follow the upload guideline", validators=[DataRequired(message="No image uploaded."), FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])
    submit = SubmitField('Upload')
