// The code here returns an alert div in the event an error has occured like upload errors, etc.

import React from 'react';
import {Alert} from 'react-bootstrap';

const MessageAlert = ({type={info}, text,onClose}) => {
    if (!text) {return null};

    return (
        <Alert key={type} variant={type} onClose={onClose} dismissible>
        {text}
        </Alert>
    );
};

export default MessageAlert;