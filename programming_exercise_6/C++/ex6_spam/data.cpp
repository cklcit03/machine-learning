// Copyright (C) 2015  Caleb Lo
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.

// DataEmail class functions convert an e-mail in a text file into training 
// data.

#include "data.h"

// Performs non-trivial initialization.
DataEmail::DataEmail(std::string email_file_name_arg) {
  std::vector<char> file_contents;
  FILE *train_file_pointer = fopen(email_file_name_arg.c_str(),"r");
  char *fileChar = new char [1];
  while(!feof(train_file_pointer)) {
    int read_file_result = fread(fileChar,sizeof(char),1,train_file_pointer);
    assert(read_file_result >= 0);
    file_contents.push_back(*fileChar);
  }
  fclose(train_file_pointer);

  // Extracts features from file.
  std::vector<std::string> vocab_list;
  const int kReturnCode = GetVocabList(&vocab_list);
  std::vector<char> file_contents_lower;
  std::string file_contents_lower_string = "";
  for(int vec_idx=0; vec_idx<(int)file_contents.size(); vec_idx++)
  {
    std::string curr_string = "";
    char curr_char = tolower(file_contents.at(vec_idx));
    if (curr_char == '\n') {
      file_contents_lower.push_back(' ');
      curr_string.push_back(' ');
    }
    else {
      file_contents_lower.push_back(curr_char);
      curr_string.push_back(curr_char);
    }
    file_contents_lower_string = file_contents_lower_string+curr_string;
  }
  std::regex strip_html("<[^<>]+>");
  std::string no_html;
  std::regex_replace(std::back_inserter(no_html),\
    file_contents_lower_string.begin(),file_contents_lower_string.end(),\
    strip_html,std::string(" "));
  std::regex strip_numbers("[0-9]+");
  std::string no_numbers;
  std::regex_replace(std::back_inserter(no_numbers),no_html.begin(),\
    no_html.end(),strip_numbers,std::string("number"));
  std::regex strip_urls("(http|https)://[^ ]+");
  std::string no_urls;
  std::regex_replace(std::back_inserter(no_urls),no_numbers.begin(),\
    no_numbers.end(),strip_urls,std::string("httpaddr"));
  std::regex strip_emails("[^ ]+@[^ ]+");
  std::string no_emails;
  std::regex_replace(std::back_inserter(no_emails),no_urls.begin(),\
    no_urls.end(),strip_emails,std::string("emailaddr"));
  std::regex strip_dollar_signs("[$]+");
  std::string no_dollar_signs;
  std::regex_replace(std::back_inserter(no_dollar_signs),no_emails.begin(),\
    no_emails.end(),strip_dollar_signs,std::string("dollar"));
  printf("==== Processed Email ====\n");

  // Removes punctuation in processed e-mail.
  std::regex strip_punctuation("[[:punct:]]");
  std::string no_punctuation;
  std::regex_replace(std::back_inserter(no_punctuation),\
    no_dollar_signs.begin(),no_dollar_signs.end(),strip_punctuation,\
    std::string(""));
  std::istringstream no_punct_stream(no_punctuation);
  std::vector<std::string> words_mod_file_contents;
  std::copy(std::istream_iterator<std::string>(no_punct_stream),\
    std::istream_iterator<std::string>(),\
    std::back_inserter(words_mod_file_contents));
  std::regex strip_non_alphanumerics("[^ ]+@[^ ]+");
  for(int word_idx=0; word_idx<(int)words_mod_file_contents.size(); \
    word_idx++)
  {
    std::string curr_word;
    std::regex_replace(std::back_inserter(curr_word),\
      words_mod_file_contents.at(word_idx).begin(),\
      words_mod_file_contents.at(word_idx).end(),strip_non_alphanumerics,\
      std::string(""));

    // Applies Porter2 stemming algorithm from Sean Massung.
    Porter2Stemmer::stem(curr_word);
    if (curr_word.length() > 1) {

      // Searches through vocab_list for stemmed word.
      for(int vocab_idx=0; vocab_idx<(int)vocab_list.size(); vocab_idx++)
      {
        if (curr_word.compare(vocab_list.at(vocab_idx)) == 0) {
          word_indices_.push_back(vocab_idx);
        }
      }

      // Displays stemmed word.
      std::cout << curr_word << "\n";
    }
  }
  printf("=========================\n");

  // Processes list of word indices and sets features.
  const int kNumDictionaryWords = 1899;
  features_.zeros(1,kNumDictionaryWords);
  for(int word_idx=0; word_idx<(int)word_indices_.size(); word_idx++)
  {
    int curr_feat_idx = word_indices_.at(word_idx);
    features_.at(0,curr_feat_idx) = 1.0;
  }
  set_num_features(kNumDictionaryWords);

  delete [] fileChar;
}

// The file that this function reads has two columns.
// One column is a list of word indices.
// The other column is a list of words.
int DataEmail::GetVocabList(std::vector<std::string> *vocab_file_words)
{
  std::string curr_line;
  std::ifstream vocab_file("../../vocab.txt");
  assert(vocab_file.is_open());
  while (std::getline(vocab_file,curr_line)) {

    // Splits current line into tokens and only saves the second token.
    // The second token corresponds to the word on the current line.
    char *token_index = strtok((char *)curr_line.c_str(),"\t");
    char *token_word = strtok(NULL," ");
    vocab_file_words->push_back(token_word);
  }
  vocab_file.close();

  return 0;
}
